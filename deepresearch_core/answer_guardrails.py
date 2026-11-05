from typing import Any

from deepresearch_core.standalone.llm import redact_text


ANSWER_GUARDRAILS_SCHEMA_VERSION = 1
ANSWER_GUARDRAIL_ITEM_LIMIT = 4
ANSWER_GUARDRAIL_TEXT_LIMIT = 180


def normalize_answer_guardrails(
    value: Any = None,
    *,
    provenance: Any = None,
    source_quality: Any = None,
    answer_evidence: Any = None,
) -> dict[str, Any]:
    """Return a compact answer-use guardrail from provenance and source signals."""

    raw = value if isinstance(value, dict) else {}
    provenance_data = provenance if isinstance(provenance, dict) else {}
    quality_data = source_quality if isinstance(source_quality, dict) else {}
    evidence_data = answer_evidence if isinstance(answer_evidence, dict) else {}
    claims = _claim_items(provenance_data)
    missing = [claim for claim in claims if _is_missing_claim(claim)]
    contradicted = [claim for claim in claims if claim.get("provider_verdict") == "contradicted"]
    unsupported = [claim for claim in claims if claim.get("provider_verdict") == "unsupported" or _is_missing_claim(claim)]
    uncertain = [claim for claim in claims if claim.get("provider_verdict") == "uncertain"]
    items = _guardrail_items(contradicted, unsupported, uncertain)
    status, label, action, public_notice, rewrite_policy = _guardrail_status(
        contradicted_count=len(contradicted),
        unsupported_count=len(unsupported),
        uncertain_count=len(uncertain),
        missing_count=len(missing),
        source_quality_level=str(quality_data.get("level") or ""),
        answer_evidence_level=str(evidence_data.get("level") or ""),
    )
    return {
        "schema_version": ANSWER_GUARDRAILS_SCHEMA_VERSION,
        "status": _choice(raw.get("status"), {"clear", "caution", "blocked", "missing"}, status),
        "label": _compact(raw.get("label"), limit=40) or label,
        "action": _choice(raw.get("action"), {"read_with_sources", "repair_evidence", "resolve_conflicts", "add_sources"}, action),
        "public_notice": _compact(raw.get("public_notice"), limit=140) or public_notice,
        "rewrite_policy": _compact(raw.get("rewrite_policy"), limit=140) or rewrite_policy,
        "claim_count": _safe_count(raw.get("claim_count")) if raw.get("claim_count") is not None else len(claims),
        "guarded_claim_count": _safe_count(raw.get("guarded_claim_count")) if raw.get("guarded_claim_count") is not None else len(items),
        "missing_claim_count": _safe_count(raw.get("missing_claim_count")) if raw.get("missing_claim_count") is not None else len(missing),
        "unsupported_claim_count": _safe_count(raw.get("unsupported_claim_count"))
        if raw.get("unsupported_claim_count") is not None
        else len(unsupported),
        "contradicted_claim_count": _safe_count(raw.get("contradicted_claim_count"))
        if raw.get("contradicted_claim_count") is not None
        else len(contradicted),
        "uncertain_claim_count": _safe_count(raw.get("uncertain_claim_count")) if raw.get("uncertain_claim_count") is not None else len(uncertain),
        "items": _normalized_items(raw.get("items")) or items,
        "note": _compact(raw.get("note"), limit=180) or "回答护栏只降低误用风险；来源、模型核验与护栏提示都不代表事实真伪。",
    }


def _guardrail_status(
    *,
    contradicted_count: int,
    unsupported_count: int,
    uncertain_count: int,
    missing_count: int,
    source_quality_level: str,
    answer_evidence_level: str,
) -> tuple[str, str, str, str, str]:
    """Return public status, label, action, notice, and rewrite policy."""

    if contradicted_count or source_quality_level == "blocked" or answer_evidence_level == "blocked":
        return (
            "blocked",
            "先核对冲突",
            "resolve_conflicts",
            "使用提示：这份回答含冲突或未解决证据信号，请先核对来源。",
            "先核对冲突来源；确认前不要把相关表述当作确定结论。",
        )
    if source_quality_level == "missing":
        return (
            "missing",
            "缺少来源",
            "add_sources",
            "使用提示：当前缺少可展示来源，请补充来源后再使用结论。",
            "没有来源时不保留事实断言；应先搜索或粘贴可信材料。",
        )
    if unsupported_count or uncertain_count or missing_count or source_quality_level == "review" or answer_evidence_level == "caution":
        return (
            "caution",
            "带条件使用",
            "repair_evidence",
            "使用提示：部分结论需补来源或复核，引用时请保留条件语气。",
            "补充来源或改写为假设；未核验前使用“可能、据当前来源、待验证”等条件表达。",
        )
    return (
        "clear",
        "可先阅读",
        "read_with_sources",
        "使用提示：可先阅读答案，关键结论仍建议打开来源核对。",
        "保留来源链接；需要引用时打开关键来源确认原文。",
    )


def _guardrail_items(
    contradicted: list[dict[str, Any]],
    unsupported: list[dict[str, Any]],
    uncertain: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return prioritized guarded claim items."""

    items: list[dict[str, Any]] = []
    for claim in contradicted:
        items.append(_guardrail_item(claim, "provider_contradicted", "冲突主张", "先核对冲突来源；确认前不要引用为确定结论。"))
    for claim in unsupported:
        if len(items) >= ANSWER_GUARDRAIL_ITEM_LIMIT:
            break
        if claim.get("provider_verdict") == "unsupported":
            issue = "provider_unsupported"
            label = "待补来源"
            guidance = "补充来源，或改写为假设/待验证表述。"
        elif _is_citation_unsupported(claim):
            issue = "citation_unsupported"
            label = "引用未支撑"
            guidance = "不要把该引用当作依据；请补充能支撑该主张的来源，或改写为假设/待验证表述。"
        else:
            issue = "missing_source"
            label = "待补来源"
            guidance = "补充来源，或改写为假设/待验证表述。"
        items.append(_guardrail_item(claim, issue, label, guidance))
    for claim in uncertain:
        if len(items) >= ANSWER_GUARDRAIL_ITEM_LIMIT:
            break
        items.append(_guardrail_item(claim, "provider_uncertain", "需人工复核", "保留条件语气，并打开来源人工核对。"))
    return _dedupe_items(items)[:ANSWER_GUARDRAIL_ITEM_LIMIT]


def _guardrail_item(claim: dict[str, Any], issue: str, label: str, guidance: str) -> dict[str, Any]:
    """Return one safe guarded claim item."""

    return {
        "claim_id": _compact(claim.get("id"), limit=24) or "claim",
        "issue": issue,
        "label": label,
        "text": _compact(claim.get("text"), limit=ANSWER_GUARDRAIL_TEXT_LIMIT),
        "guidance": guidance,
    }


def _claim_items(provenance: dict[str, Any]) -> list[dict[str, Any]]:
    """Return public claim dicts from provenance."""

    claims = provenance.get("claims")
    if not isinstance(claims, list):
        return []
    return [claim for claim in claims if isinstance(claim, dict)]


def _is_missing_claim(claim: dict[str, Any]) -> bool:
    """Return whether a claim has no mapped source support."""

    return str(claim.get("status") or "") == "missing" or str(claim.get("support_level") or "") == "unsupported"


def _is_citation_unsupported(claim: dict[str, Any]) -> bool:
    """Return whether a mapped citation exists but fails the support precheck."""

    citation_support = claim.get("citation_support") if isinstance(claim.get("citation_support"), dict) else {}
    return (
        str(claim.get("status") or "") == "mapped"
        and str(claim.get("support_level") or "") == "unsupported"
        and str(citation_support.get("level") or "") == "unsupported"
    )


def _normalized_items(value: Any) -> list[dict[str, Any]]:
    """Return safe user-supplied items when present."""

    if not isinstance(value, list):
        return []
    items = []
    for item in value[:ANSWER_GUARDRAIL_ITEM_LIMIT]:
        if not isinstance(item, dict):
            continue
        claim_id = _compact(item.get("claim_id"), limit=24)
        text = _compact(item.get("text"), limit=ANSWER_GUARDRAIL_TEXT_LIMIT)
        if not claim_id and not text:
            continue
        items.append(
            {
                "claim_id": claim_id or "claim",
                "issue": _choice(
                    item.get("issue"),
                    {"missing_source", "citation_unsupported", "provider_unsupported", "provider_contradicted", "provider_uncertain", "source_quality"},
                    "missing_source",
                ),
                "label": _compact(item.get("label"), limit=40) or "待复核",
                "text": text,
                "guidance": _compact(item.get("guidance"), limit=120) or "补充来源或保留条件语气。",
            }
        )
    return items


def _dedupe_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return items deduplicated by claim id and issue."""

    seen = set()
    result = []
    for item in items:
        key = (item["claim_id"], item["issue"])
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


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


def _compact(value: Any, *, limit: int) -> str:
    """Return redacted single-line text."""

    text = redact_text(str(value or ""))
    text = " ".join(text.split()).strip()
    return text[: max(0, int(limit))]
