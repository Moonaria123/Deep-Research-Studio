from typing import Any

from deepresearch_core.report_sources import sanitize_report_sources
from deepresearch_core.standalone.llm import redact_text


ANSWER_EVIDENCE_SCHEMA_VERSION = 4
ANSWER_EVIDENCE_PRIMARY_SOURCE_LIMIT = 3
ANSWER_EVIDENCE_SOURCE_SCAN_LIMIT = 12
ANSWER_EVIDENCE_TEXT_LIMIT = 220
ANSWER_EVIDENCE_PROMPT_LIMIT = 260
ANSWER_EVIDENCE_REPAIR_ACTION_LIMIT = 3


def normalize_answer_evidence(
    value: Any = None,
    *,
    sources: Any = None,
    confidence: Any = None,
    provenance: Any = None,
    query: Any = "",
) -> dict[str, Any]:
    """Return a compact source-first evidence summary for end users."""

    safe_sources = sanitize_report_sources(sources, limit=ANSWER_EVIDENCE_SOURCE_SCAN_LIMIT)
    confidence_data = confidence if isinstance(confidence, dict) else {}
    provenance_data = provenance if isinstance(provenance, dict) else {}
    raw = value if isinstance(value, dict) else {}
    source_count = len(safe_sources)
    conflict_gate = provenance_data.get("conflict_gate") if isinstance(provenance_data.get("conflict_gate"), dict) else {}
    evidence_repair = provenance_data.get("evidence_repair") if isinstance(provenance_data.get("evidence_repair"), dict) else {}
    quality_gate = provenance_data.get("quality_gate") if isinstance(provenance_data.get("quality_gate"), dict) else {}
    source_relevance = provenance_data.get("source_relevance") if isinstance(provenance_data.get("source_relevance"), dict) else {}
    semantic = provenance_data.get("semantic_verification") if isinstance(provenance_data.get("semantic_verification"), dict) else {}
    ranked_sources = _ranked_sources(safe_sources, source_relevance)
    primary_sources = _primary_sources(ranked_sources)

    level, label, action, summary, risk_count = _evidence_status(
        source_count=source_count,
        conflict_gate=conflict_gate,
        evidence_repair=evidence_repair,
        quality_gate=quality_gate,
        source_relevance=source_relevance,
        semantic=semantic,
        confidence=confidence_data,
    )
    raw_summary = _compact(raw.get("summary"))
    raw_note = _compact(raw.get("note"))
    repair_actions = _repair_actions(
        raw.get("repair_actions") or raw.get("repairActions"),
        query=query,
        source_count=source_count,
        conflict_gate=conflict_gate,
        evidence_repair=evidence_repair,
        source_relevance=source_relevance,
    )
    return {
        "schema_version": ANSWER_EVIDENCE_SCHEMA_VERSION,
        "level": _choice(raw.get("level"), {"ready", "caution", "blocked", "unverified"}, level),
        "label": _compact(raw.get("label")) or label,
        "action": _compact(raw.get("action")) or action,
        "source_count": source_count,
        "risk_count": risk_count,
        "primary_sources": primary_sources,
        "ranked_sources": ranked_sources,
        "repair_actions": repair_actions,
        "summary": raw_summary or summary,
        "note": raw_note or "该摘要只说明来源、引用和冲突核验状态，不代表事实真伪。",
    }


def _evidence_status(
    *,
    source_count: int,
    conflict_gate: dict[str, Any],
    evidence_repair: dict[str, Any],
    quality_gate: dict[str, Any],
    source_relevance: dict[str, Any],
    semantic: dict[str, Any],
    confidence: dict[str, Any],
) -> tuple[str, str, str, str, int]:
    """Choose the most user-relevant source-first status."""

    conflict_items = _item_count(conflict_gate)
    repair_items = _item_count(evidence_repair)
    if str(conflict_gate.get("level") or "") == "conflict":
        risk_count = max(1, conflict_items)
        return (
            "blocked",
            "先核对冲突",
            "resolve_conflicts_before_using",
            f"冲突门禁发现 {risk_count} 个需优先核对的信号；先看来源并核对冲突，再决定是否引用。",
            risk_count,
        )
    if str(evidence_repair.get("level") or "") == "conflict":
        risk_count = max(1, repair_items)
        return (
            "blocked",
            "先处理证据冲突",
            "resolve_conflicts_before_using",
            f"证据修复队列发现 {risk_count} 个冲突相关动作；先看来源并处理冲突，再使用回答。",
            risk_count,
        )
    if source_count <= 0:
        return (
            "unverified",
            "缺少可展示来源",
            "add_sources_or_search",
            "没有可展示来源；我不会建议直接引用这段回答，请先补充来源或重新搜索。",
            max(conflict_items, repair_items),
        )
    source_level = str(source_relevance.get("level") or "")
    repair_level = str(evidence_repair.get("level") or "")
    quality_level = str(quality_gate.get("level") or "")
    semantic_level = str(semantic.get("level") or "")
    semantic_claim_count = len(semantic.get("claim_ids") or []) if isinstance(semantic.get("claim_ids"), list) else 0
    if (
        str(conflict_gate.get("level") or "") == "caution"
        or source_level in {"missing", "unrelated", "weak"}
        or repair_level == "needs_repair"
        or quality_level == "review_required"
        or (semantic_claim_count and semantic_level in {"needs_evidence", "needs_review"})
    ):
        risk_count = max(conflict_items, repair_items, len(quality_gate.get("claim_ids") or []), len(semantic.get("claim_ids") or []))
        return (
            "caution",
            "先核对证据",
            "repair_evidence",
            f"先看来源再阅读正文；当前有 {max(1, risk_count)} 个证据或核验信号建议复核。",
            max(1, risk_count),
        )
    confidence_level = str(confidence.get("level") or "")
    label = "可带来源阅读" if confidence_level != "low" else "谨慎带来源阅读"
    return (
        "ready",
        label,
        "read_with_sources",
        f"先看来源再阅读正文；当前有 {source_count} 个可展示来源，未发现需要优先阻断的冲突信号。",
        0,
    )


def _ranked_sources(sources: list[dict[str, Any]], source_relevance: dict[str, Any]) -> list[dict[str, Any]]:
    """Return ranked source hints for the answer evidence capsule."""

    relevance_items = source_relevance.get("items") if isinstance(source_relevance.get("items"), list) else []
    relevance_by_url: dict[str, dict[str, Any]] = {}
    for item in relevance_items:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url") or "").strip()
        if url:
            relevance_by_url.setdefault(url, item)

    scored: list[dict[str, Any]] = []
    for index, source in enumerate(sources, start=1):
        url = str(source.get("url") or "").strip()
        relevance = relevance_by_url.get(url, {})
        score = _bounded_score(relevance.get("score"))
        level = _rank_level(relevance.get("level"), score)
        priority = {"relevant": 3, "weak": 2, "unknown": 1, "unrelated": 0}.get(level, 0)
        scored.append(
            {
                "score": score,
                "priority": priority,
                "index": index,
                "item": _ranked_source_item(source, relevance, level=level, score=score),
            }
        )

    ranked = []
    sorted_sources = sorted(scored, key=lambda row: (-row["score"], -row["priority"], row["index"]))
    for rank, row in enumerate(sorted_sources[:ANSWER_EVIDENCE_PRIMARY_SOURCE_LIMIT], start=1):
        item = row["item"]
        item["rank"] = rank
        item["reason"] = _rank_reason(rank, item["relevance_level"], item["relevance_score"], item["matched_terms"])
        ranked.append(item)
    return ranked


def _repair_actions(
    value: Any,
    *,
    query: Any,
    source_count: int,
    conflict_gate: dict[str, Any],
    evidence_repair: dict[str, Any],
    source_relevance: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return safe follow-up prompts for repairing answer-level evidence."""

    raw_actions = _raw_repair_actions(value)
    if raw_actions:
        return raw_actions

    actions: list[dict[str, Any]] = []
    if str(conflict_gate.get("level") or "") == "conflict":
        for item in _safe_items(conflict_gate):
            actions.append(
                _repair_action_item(
                    kind=item.get("kind") or "conflict_gate",
                    claim_id=item.get("claim_id"),
                    priority=1,
                    action=item.get("action") or "resolve_conflicts_before_using",
                    label="核对冲突来源",
                    prompt=_repair_prompt("请优先核对冲突来源后再继续回答", query, item.get("text") or conflict_gate.get("note")),
                )
            )

    for item in _safe_items(evidence_repair):
        if len(actions) >= ANSWER_EVIDENCE_REPAIR_ACTION_LIMIT:
            break
        kind = _compact(item.get("kind"), limit=40) or "evidence_repair"
        actions.append(
            _repair_action_item(
                kind=kind,
                claim_id=item.get("claim_id"),
                priority=item.get("priority"),
                action=item.get("action") or _repair_action(kind),
                label=_repair_label(kind),
                prompt=_repair_prompt(_repair_prompt_prefix(kind), query, item.get("text") or evidence_repair.get("note")),
            )
        )

    source_level = str(source_relevance.get("level") or "")
    if not actions and (source_count <= 0 or source_level in {"missing", "unrelated", "weak"}):
        kind = "missing_sources" if source_count <= 0 or source_level == "missing" else "source_relevance"
        actions.append(
            _repair_action_item(
                kind=kind,
                claim_id="",
                priority=1,
                action=source_relevance.get("action") or "refine_search_or_add_sources",
                label=_repair_label(kind),
                prompt=_repair_prompt(_repair_prompt_prefix(kind), query, source_relevance.get("note")),
            )
        )

    return [action for action in actions if action.get("prompt")][:ANSWER_EVIDENCE_REPAIR_ACTION_LIMIT]


def _raw_repair_actions(value: Any) -> list[dict[str, Any]]:
    """Return caller-provided repair actions after public-field sanitization."""

    if not isinstance(value, list):
        return []
    actions: list[dict[str, Any]] = []
    for item in value[:ANSWER_EVIDENCE_REPAIR_ACTION_LIMIT]:
        if not isinstance(item, dict):
            continue
        prompt = _compact(item.get("prompt"), limit=ANSWER_EVIDENCE_PROMPT_LIMIT)
        if not prompt:
            continue
        kind = _compact(item.get("kind"), limit=40) or "custom"
        actions.append(
            _repair_action_item(
                kind=kind,
                claim_id=item.get("claim_id") or item.get("claimId"),
                priority=item.get("priority"),
                action=item.get("action") or _repair_action(kind),
                label=item.get("label") or _repair_label(kind),
                prompt=prompt,
            )
        )
    return actions


def _safe_items(value: dict[str, Any]) -> list[dict[str, Any]]:
    """Return dict items from a public provenance block."""

    items = value.get("items")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def _repair_action_item(
    *,
    kind: Any,
    claim_id: Any,
    priority: Any,
    action: Any,
    label: Any,
    prompt: Any,
) -> dict[str, Any]:
    """Return one sanitized answer-evidence repair action."""

    return {
        "kind": _compact(kind, limit=40) or "evidence_repair",
        "claim_id": _compact(claim_id, limit=24),
        "priority": _bounded_priority(priority),
        "label": _compact(label, limit=48) or "补充证据",
        "action": _compact(action, limit=80) or "repair_evidence",
        "prompt": _compact(prompt, limit=ANSWER_EVIDENCE_PROMPT_LIMIT),
        "requires_confirmation": True,
        "confirmation_label": "开始补证",
        "confirmation_note": "需要你确认后才会启动本地研究流程，可能使用本机 BYOK 配置。",
    }


def _repair_label(kind: str) -> str:
    """Return a compact user-facing label for an evidence action kind."""

    return {
        "source_relevance": "补充相关来源",
        "missing_sources": "重新搜索来源",
        "provider_action": "核对模型核验",
        "citation_support": "补充支撑来源",
        "review_action": "补充可核验来源",
        "semantic_review": "逐句核对来源",
        "conflict_gate": "核对冲突来源",
    }.get(kind, "补充证据")


def _repair_action(kind: str) -> str:
    """Return a stable public action id for an evidence action kind."""

    return {
        "source_relevance": "refine_search_or_add_sources",
        "missing_sources": "add_sources_or_search",
        "provider_action": "review_provider_verification",
        "citation_support": "add_supporting_source_or_rewrite_claim",
        "review_action": "add_sources_or_rewrite_claim",
        "semantic_review": "open_sources_to_verify",
        "conflict_gate": "resolve_conflicts_before_using",
    }.get(kind, "repair_evidence")


def _repair_prompt_prefix(kind: str) -> str:
    """Return the safe prompt prefix for an evidence action kind."""

    return {
        "source_relevance": "请补充更相关来源后再回答",
        "missing_sources": "请重新搜索并补充可核验来源后再回答",
        "provider_action": "请先核对模型核验提示后再继续回答",
        "citation_support": "请补充能支撑该主张的来源后再继续回答",
        "review_action": "请补充可核验来源后再继续回答",
        "semantic_review": "请打开来源逐句比对后再继续回答",
        "conflict_gate": "请优先核对冲突来源后再继续回答",
    }.get(kind, "请补充证据后再继续回答")


def _repair_prompt(prefix: str, query: Any, detail: Any) -> str:
    """Return a bounded follow-up prompt without local paths or secret-like text."""

    safe_prefix = _compact(prefix, limit=80) or "请补充证据后再继续回答"
    safe_query = _compact(query, limit=80)
    safe_detail = _compact(detail, limit=150)
    query_part = f"与“{safe_query}”直接相关的" if safe_query else ""
    detail_part = f"：{safe_detail}" if safe_detail else "。"
    return _compact(f"{safe_prefix.replace('请补充', f'请补充{query_part}', 1) if query_part and safe_prefix.startswith('请补充') else safe_prefix} {detail_part}", limit=ANSWER_EVIDENCE_PROMPT_LIMIT)


def _bounded_priority(value: Any) -> int:
    """Return a bounded 1-9 action priority."""

    try:
        priority = int(value)
    except (TypeError, ValueError):
        priority = 3
    return max(1, min(9, priority))


def _ranked_source_item(source: dict[str, Any], relevance: dict[str, Any], *, level: str, score: int) -> dict[str, Any]:
    """Return one compact ranked source item."""

    return {
        "rank": 0,
        "title": _compact(source.get("title")) or _compact(source.get("url")),
        "url": str(source.get("url") or "").strip(),
        "site": _compact(source.get("site")),
        "relevance_score": score,
        "relevance_level": level,
        "matched_terms": _matched_terms(relevance.get("matched_terms")),
        "reason": "",
    }


def _primary_sources(sources: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return compact public source links for the evidence capsule."""

    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for source in sources[:ANSWER_EVIDENCE_PRIMARY_SOURCE_LIMIT]:
        url = str(source.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        items.append(
            {
                "title": _compact(source.get("title")) or url,
                "url": url,
                "site": _compact(source.get("site")),
            }
        )
    return items


def _rank_reason(rank: int, level: str, score: int, matched_terms: list[str]) -> str:
    """Return a short source ranking explanation."""

    terms = "、".join(matched_terms[:3])
    if level == "relevant":
        suffix = f"匹配 {terms}。" if terms else "与问题关键词匹配度较高。"
        prefix = "首要来源" if rank == 1 else "优先来源"
        return f"{prefix}：相关性 {score}/100，{suffix}"
    if level == "weak":
        suffix = f"仅匹配 {terms}。" if terms else "覆盖部分关键词。"
        return f"候选来源：相关性 {score}/100，{suffix}请打开核对。"
    if level == "unknown":
        return "候选来源：缺少可解释相关性，请打开核对。"
    return f"低相关来源：相关性 {score}/100，仅作线索，不建议直接引用。"


def _rank_level(value: Any, score: int) -> str:
    """Return one public source ranking level."""

    text = str(value or "").strip().lower()
    if text in {"relevant", "weak", "unknown", "unrelated"}:
        return text
    if score >= 35:
        return "relevant"
    if score >= 18:
        return "weak"
    return "unrelated"


def _matched_terms(value: Any) -> list[str]:
    """Return compact safe matched terms."""

    if not isinstance(value, list):
        return []
    terms = []
    seen = set()
    for item in value:
        term = _compact(item, limit=48).lower()
        if not term or term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= 8:
            break
    return terms


def _bounded_score(value: Any) -> int:
    """Return a bounded 0-100 score."""

    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 0
    return max(0, min(100, score))


def _item_count(value: dict[str, Any]) -> int:
    """Return a public count of risk/action items without trusting raw types."""

    items = value.get("items")
    if isinstance(items, list) and items:
        return len(items)
    claim_ids = value.get("claim_ids")
    if isinstance(claim_ids, list):
        return len(claim_ids)
    return 0


def _choice(value: Any, allowed: set[str], fallback: str) -> str:
    """Return one allowed string value."""

    text = _compact(value)
    return text if text in allowed else fallback


def _compact(value: Any, *, limit: int = ANSWER_EVIDENCE_TEXT_LIMIT) -> str:
    """Return redacted single-line text."""

    text = " ".join(redact_text(str(value or "")).split()).strip()
    return text[:limit]
