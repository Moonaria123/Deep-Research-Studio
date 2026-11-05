import re
from typing import Any, Iterable

from deepresearch_core.report_sources import sanitize_report_sources
from deepresearch_core.source_relevance import build_source_relevance, normalize_source_relevance
from deepresearch_core.standalone.llm import redact_text


JsonDict = dict[str, Any]
DEFAULT_CLAIM_LIMIT = 8
CLAIM_TEXT_LIMIT = 220
REVIEW_ACTION_TEXT_LIMIT = 180
PROVIDER_ACTION_TEXT_LIMIT = 180
CONFLICT_GATE_TEXT_LIMIT = 200
EVIDENCE_REPAIR_TEXT_LIMIT = 200
QUALITY_GATE_NOTE_LIMIT = 240
SEMANTIC_VERIFICATION_NOTE_LIMIT = 260
PROVIDER_VERIFICATION_NOTE_LIMIT = 260
PROVENANCE_SCHEMA_VERSION = 11
SUPPORT_SCORE_STRONG = 100
SUPPORT_SCORE_LINKED = 75
SEMANTIC_SCORE_READY = 70
SEMANTIC_SCORE_LINKED = 45
CITATION_SUPPORT_TEXT_LIMIT = 4200
CITATION_SUPPORT_MIN_SCORE = 24
_SUPPORT_TOKEN_RE = re.compile(r"(?i)[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*|\d+(?:\.\d+)?|[\u4e00-\u9fff]{2,}")
_SUPPORT_CJK_RE = re.compile(r"^[\u4e00-\u9fff]+$")
_SUPPORT_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")
_SUPPORT_STOP_TERMS = frozenset(
    {
        "什么",
        "怎么",
        "如何",
        "是否",
        "有没有",
        "多少",
        "是谁",
        "这个",
        "那个",
        "一下",
        "请问",
        "帮我",
        "需要",
        "可以",
        "使用",
        "来源",
        "核验",
        "支撑",
        "the",
        "a",
        "an",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "what",
        "how",
        "who",
        "when",
        "requires",
        "require",
        "required",
        "before",
        "after",
        "with",
        "without",
        "source",
        "sources",
        "backed",
        "verification",
        "verify",
        "use",
        "uses",
        "using",
        "should",
    }
)
PROVIDER_VERDICT_LABELS = {
    "supported": "模型认为来源支持",
    "uncertain": "模型无法确认",
    "contradicted": "模型发现冲突",
    "unsupported": "模型未找到支撑",
    "not_checked": "未启用模型核验",
}
PROVIDER_VERDICT_NOTES = {
    "supported": "模型判断该主张与已映射来源语义一致；仍建议打开来源核对。",
    "uncertain": "模型无法基于已映射来源确认该主张；引用前请人工复核。",
    "contradicted": "模型发现该主张与已映射来源可能冲突；引用前请核对原文。",
    "unsupported": "模型未在已映射来源中找到充分支撑；请补充来源或改写为假设。",
}
PROVIDER_VERDICTS = frozenset(PROVIDER_VERDICT_LABELS)
PROVIDER_ACTIONS = {
    "contradicted": "核对冲突来源",
    "unsupported": "补充来源或改写为假设",
    "uncertain": "人工复核",
}
_CITATION_RE = re.compile(r"\[(\d{1,2})\]")
_LOCAL_PATH_RE = re.compile(
    r"(?i)file:///[^\r\n,;)]*|"
    r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/][^\r\n,;)]*|"
    r"(?<![A-Za-z0-9_])(?:/home|/users|/var|/tmp|/mnt|/workspace|/root|/etc|/opt|/private)/[^\r\n,;)]*",
)


def empty_claim_provenance(*, provider_verification: Any = None, query: Any = "", sources: Iterable[dict[str, Any]] | None = None) -> JsonDict:
    """Return an empty public claim provenance object."""

    return _provenance_from_claims([], provider_verification=provider_verification, source_relevance=build_source_relevance(query, sources))


def build_claim_provenance(
    markdown: Any,
    sources: Iterable[dict[str, Any]] | None,
    *,
    limit: int = DEFAULT_CLAIM_LIMIT,
    provider_verification: Any = None,
    query: Any = "",
) -> JsonDict:
    """Build a safe claim-to-source map from Markdown body citations."""

    raw_sources = list(sources or [])
    safe_sources = sanitize_report_sources(raw_sources)
    raw_sources_by_url = _raw_sources_by_safe_url(raw_sources)
    claims = []
    for text in _claim_paragraphs(markdown):
        if len(claims) >= _safe_limit(limit):
            break
        source_indexes = _citation_indexes(text, len(safe_sources))
        claims.append(_claim_item(len(claims) + 1, text, source_indexes, safe_sources, raw_sources_by_url))
    return _provenance_from_claims(
        claims,
        provider_verification=provider_verification,
        source_relevance=build_source_relevance(query, raw_sources),
    )


def normalize_report_provenance(
    value: Any,
    *,
    markdown: Any = "",
    sources: Iterable[dict[str, Any]] | None = None,
    limit: int = DEFAULT_CLAIM_LIMIT,
    provider_verification: Any = None,
    query: Any = "",
) -> JsonDict:
    """Return safe provenance metadata, deriving it from Markdown when absent."""

    raw_sources = list(sources or [])
    safe_sources = sanitize_report_sources(raw_sources)
    raw_sources_by_url = _raw_sources_by_safe_url(raw_sources)
    if isinstance(value, dict) and isinstance(value.get("claims"), list):
        provider_payload = provider_verification if provider_verification is not None else value.get("provider_verification")
        source_relevance = normalize_source_relevance(value.get("source_relevance"), query=query, sources=raw_sources)
        claims = []
        for item in value.get("claims", [])[: _safe_limit(limit)]:
            if not isinstance(item, dict):
                continue
            text = _safe_text(item.get("text"), CLAIM_TEXT_LIMIT)
            if not text:
                continue
            source_indexes = _normalize_source_indexes(item, len(safe_sources))
            claim = _claim_item(len(claims) + 1, text, source_indexes, safe_sources, raw_sources_by_url)
            if _existing_citation_support_unsupported(item):
                claim.update(_citation_unsupported_claim_fields(claim.get("citation_support")))
            claim.update(_existing_provider_claim_fields(item))
            claims.append(claim)
        if claims or isinstance(value.get("source_relevance"), dict):
            return _provenance_from_claims(claims, provider_verification=provider_payload, source_relevance=source_relevance)
    if str(markdown or "").strip():
        return build_claim_provenance(markdown, raw_sources, limit=limit, provider_verification=provider_verification, query=query)
    return empty_claim_provenance(provider_verification=provider_verification, query=query, sources=raw_sources)


def _provenance_from_claims(
    claims: list[JsonDict],
    *,
    provider_verification: Any = None,
    source_relevance: Any = None,
) -> JsonDict:
    """Return the public provenance envelope for sanitized claim items."""

    provider_summary, claims = _claims_with_provider_verification(claims, provider_verification)
    mapped = sum(1 for item in claims if item.get("status") == "mapped")
    missing = len(claims) - mapped
    support_summary = _support_summary(claims)
    review_actions = _review_actions(claims)
    provider_actions = _provider_actions(claims, provider_summary)
    quality_gate = _quality_gate(claims, support_summary, review_actions)
    source_relevance_summary = normalize_source_relevance(source_relevance)
    semantic_verification = _semantic_verification(claims)
    conflict_gate = _conflict_gate(
        claims=claims,
        source_relevance=source_relevance_summary,
        provider_summary=provider_summary,
        provider_actions=provider_actions,
    )
    evidence_repair = _evidence_repair(
        claims=claims,
        quality_gate=quality_gate,
        source_relevance=source_relevance_summary,
        semantic_verification=semantic_verification,
        provider_actions=provider_actions,
        review_actions=review_actions,
    )
    return {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "claim_count": len(claims),
        "mapped_claim_count": mapped,
        "missing_claim_count": missing,
        "coverage": _coverage(len(claims), mapped, missing),
        "support_summary": support_summary,
        "quality_gate": quality_gate,
        "source_relevance": source_relevance_summary,
        "semantic_verification": semantic_verification,
        "provider_verification": provider_summary,
        "provider_actions": provider_actions,
        "conflict_gate": conflict_gate,
        "evidence_repair": evidence_repair,
        "review_actions": review_actions,
        "claims": claims,
    }


def _claim_item(
    index: int,
    text: Any,
    source_indexes: list[int],
    sources: list[JsonDict],
    raw_sources_by_url: dict[str, JsonDict],
) -> JsonDict:
    """Return one sanitized claim provenance item."""

    safe_text = _safe_text(text, CLAIM_TEXT_LIMIT)
    support = _claim_support(safe_text, source_indexes, sources, raw_sources_by_url)
    semantic = _claim_semantic_verification(source_indexes, support)
    return {
        "id": f"C{index}",
        "text": safe_text,
        "status": "mapped" if source_indexes else "missing",
        "source_indexes": source_indexes,
        "sources": [_source_ref(source_index, sources[source_index - 1]) for source_index in source_indexes],
        "support_score": support["score"],
        "support_level": support["level"],
        "support_label": support["label"],
        "citation_support": support["citation_support"],
        "semantic_score": semantic["score"],
        "semantic_level": semantic["level"],
        "semantic_label": semantic["label"],
        "semantic_note": semantic["note"],
        "provider_verdict": "not_checked",
        "provider_score": 0,
        "provider_label": PROVIDER_VERDICT_LABELS["not_checked"],
        "provider_note": "默认离线；未执行 provider-backed 模型语义核验。",
    }


def _source_ref(index: int, source: JsonDict) -> JsonDict:
    """Return a compact source reference for one claim."""

    return {
        "index": index,
        "title": _safe_text(source.get("title") or source.get("url") or "", 180),
        "url": _safe_text(source.get("url") or "", 512),
        "site": _safe_text(source.get("site") or source.get("source") or "", 120),
    }


def _existing_citation_support_unsupported(item: JsonDict) -> bool:
    """Return whether existing provenance already conservatively marked citation mismatch."""

    citation_support = item.get("citation_support")
    citation_level = str(citation_support.get("level") if isinstance(citation_support, dict) else "").strip().lower()
    support_level = str(item.get("support_level") or item.get("supportLevel") or "").strip().lower()
    return citation_level == "unsupported" or support_level == "unsupported"


def _citation_unsupported_claim_fields(existing: Any = None) -> JsonDict:
    """Return conservative public fields for a citation mismatch claim."""

    citation_support = existing if isinstance(existing, dict) else {}
    return {
        "support_score": 0,
        "support_level": "unsupported",
        "support_label": "引用未支撑",
        "citation_support": {
            "score": 0,
            "level": "unsupported",
            "label": "引用未支撑",
            "matched_terms": [],
            "note": _safe_text(
                citation_support.get("note")
                or "已标注来源，但本地预检未发现正文主张与来源内容的足够重合；请补充更相关来源。",
                SEMANTIC_VERIFICATION_NOTE_LIMIT,
            ),
        },
        "semantic_score": 0,
        "semantic_level": "needs_evidence",
        "semantic_label": "引用未支撑",
        "semantic_note": "已标注来源，但本地引用支撑预检未发现正文主张与来源内容的足够重合；请补充来源或改写为假设，不代表事实真伪或语义蕴含。",
    }


def _claim_paragraphs(markdown: Any) -> list[str]:
    """Return body paragraphs that are long enough to need provenance."""

    paragraphs: list[str] = []
    paragraph: list[str] = []
    code_block = False

    def flush() -> None:
        nonlocal paragraph
        text = _safe_text(" ".join(paragraph), CLAIM_TEXT_LIMIT)
        paragraph = []
        if len(_CITATION_RE.sub("", text).strip()) >= 36:
            paragraphs.append(text)

    for line in _body_lines_before_sources(markdown):
        trimmed = str(line or "").strip()
        if trimmed.startswith("```"):
            flush()
            code_block = not code_block
            continue
        if not trimmed:
            flush()
            continue
        if code_block or re.match(r"^#{1,6}\s+", trimmed):
            flush()
            continue
        if "|" in trimmed and len(trimmed) > 1:
            flush()
            continue
        paragraph.append(re.sub(r"^(?:[-*]|\d+\.)\s+", "", trimmed))
    flush()
    return paragraphs


def _body_lines_before_sources(markdown: Any) -> list[str]:
    """Return Markdown lines before the public source appendix."""

    lines = str(markdown or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")
    for index, line in enumerate(lines):
        if re.match(r"^#{2,3}\s+参考来源\s*$", line.strip()):
            return lines[:index]
    return lines


def _citation_indexes(text: str, source_count: int) -> list[int]:
    """Return unique source indexes referenced by one claim."""

    indexes: list[int] = []
    seen: set[int] = set()
    for match in _CITATION_RE.finditer(str(text or "")):
        index = int(match.group(1))
        if 1 <= index <= source_count and index not in seen:
            seen.add(index)
            indexes.append(index)
    return indexes


def _normalize_source_indexes(item: JsonDict, source_count: int) -> list[int]:
    """Return safe source indexes from untrusted provenance metadata."""

    raw_indexes = item.get("source_indexes")
    if raw_indexes is None:
        raw_indexes = item.get("sourceIndexes")
    if not isinstance(raw_indexes, list):
        raw_indexes = []
    indexes: list[int] = []
    seen: set[int] = set()
    for value in raw_indexes:
        try:
            index = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= index <= source_count and index not in seen:
            seen.add(index)
            indexes.append(index)
    return indexes


def _coverage(claim_count: int, mapped_count: int, missing_count: int) -> JsonDict:
    """Return a user-facing provenance coverage summary."""

    if claim_count <= 0 or mapped_count <= 0:
        level = "missing"
        label = "缺少主张来源"
        note = "未找到可映射到来源的正文主张；请优先打开来源核对。"
    elif missing_count == 0:
        level = "ready"
        label = "主张来源已映射"
        note = f"{mapped_count} 条主张已关联来源；关键事实仍建议打开来源核对。"
    else:
        level = "partial"
        label = "部分主张待补来源"
        note = f"{mapped_count} 条主张已关联来源，{missing_count} 条仍需补充来源。"
    return {"level": level, "label": label, "note": note}


def _claim_citation_support(
    text: str,
    source_indexes: list[int],
    sources: list[JsonDict],
    raw_sources_by_url: dict[str, JsonDict],
) -> JsonDict:
    """Return a local citation-to-claim lexical support precheck."""

    if not source_indexes:
        return {
            "score": 0,
            "level": "missing",
            "label": "缺少引用",
            "matched_terms": [],
            "note": "正文主张未标注来源；请补充来源或改写为假设。",
        }
    claim_terms = _weighted_support_terms(_CITATION_RE.sub("", text))
    if not claim_terms:
        return {
            "score": 0,
            "level": "unchecked",
            "label": "未做本地支撑预检",
            "matched_terms": [],
            "note": "正文主张缺少足够可比对关键词；仅保留引用数量信号。",
        }
    source_terms, has_comparable_text = _cited_source_terms(source_indexes, sources, raw_sources_by_url)
    if not has_comparable_text:
        return {
            "score": 0,
            "level": "unchecked",
            "label": "未做本地支撑预检",
            "matched_terms": [],
            "note": "来源缺少可比对正文或摘要；仅保留引用数量信号。",
        }
    score, matched_terms = _citation_support_score(claim_terms, source_terms)
    if score < CITATION_SUPPORT_MIN_SCORE:
        return {
            "score": score,
            "level": "unsupported",
            "label": "引用未支撑",
            "matched_terms": matched_terms,
            "note": "已标注来源，但本地预检未发现正文主张与来源内容的足够重合；请补充更相关来源。",
        }
    return {
        "score": score,
        "level": "supported",
        "label": "引用内容相关",
        "matched_terms": matched_terms,
        "note": "来源正文或摘要覆盖了主张的关键术语；仍需打开来源核对原文。",
    }


def _cited_source_terms(
    source_indexes: list[int],
    sources: list[JsonDict],
    raw_sources_by_url: dict[str, JsonDict],
) -> tuple[set[str], bool]:
    """Return searchable terms for cited source bodies and whether body text exists."""

    terms: set[str] = set()
    has_comparable_text = False
    for index in source_indexes:
        if index < 1 or index > len(sources):
            continue
        safe_source = sources[index - 1]
        raw_source = raw_sources_by_url.get(str(safe_source.get("url") or ""), {})
        comparable_parts = [
            raw_source.get("description", ""),
            raw_source.get("snippet", ""),
            raw_source.get("summary", ""),
            raw_source.get("markdown", ""),
            raw_source.get("text", ""),
            raw_source.get("content", ""),
            raw_source.get("body", ""),
        ]
        comparable_text = " ".join(str(part or "") for part in comparable_parts).strip()
        if len(comparable_text) >= 24:
            has_comparable_text = True
        parts = [
            safe_source.get("title", ""),
            safe_source.get("site", ""),
            safe_source.get("url", ""),
            raw_source.get("title", ""),
            comparable_text,
        ]
        terms.update(_support_tokens(" ".join(str(part or "") for part in parts)[:CITATION_SUPPORT_TEXT_LIMIT]))
    return terms, has_comparable_text


def _citation_support_score(claim_terms: dict[str, float], source_terms: set[str]) -> tuple[int, list[str]]:
    """Return a bounded local support score and matched terms."""

    if not claim_terms or not source_terms:
        return 0, []
    matched = [term for term in claim_terms if term in source_terms]
    total_weight = sum(claim_terms.values())
    matched_weight = sum(claim_terms[term] for term in matched)
    score = round((matched_weight / total_weight) * 100) if total_weight else 0
    numeric_terms = [term for term in claim_terms if _SUPPORT_NUMBER_RE.match(term) or any(char.isdigit() for char in term)]
    if numeric_terms and not any(term in matched for term in numeric_terms):
        score = min(score, 20)
    entity_terms = [term for term in claim_terms if term not in numeric_terms]
    if entity_terms and not any(term in matched for term in entity_terms):
        score = min(score, 15)
    if len(matched) < 2 and len(claim_terms) >= 4:
        score = min(score, 20)
    return max(0, min(100, int(score))), sorted(matched, key=lambda term: (-claim_terms[term], term))[:8]


def _weighted_support_terms(text: Any) -> dict[str, float]:
    """Return weighted public claim terms for local citation support."""

    terms: dict[str, float] = {}
    for token in _support_tokens(str(text or "")):
        terms[token] = max(terms.get(token, 0.0), _support_term_weight(token))
    return terms


def _support_tokens(text: str) -> list[str]:
    """Return compact lower-case tokens for local citation support."""

    normalized = redact_text(str(text or "")).lower()
    tokens: list[str] = []
    for raw in _SUPPORT_TOKEN_RE.findall(normalized):
        token = raw.strip("._-")
        if not token:
            continue
        if _SUPPORT_CJK_RE.match(token):
            tokens.extend(_support_cjk_tokens(token))
            continue
        tokens.append(token)
        for part in re.split(r"[-_.]+", token):
            if part and part != token:
                tokens.append(part)
    return [token for token in tokens if token and token not in _SUPPORT_STOP_TERMS]


def _support_cjk_tokens(token: str) -> list[str]:
    """Return CJK word-like tokens and bigrams for local citation support."""

    if len(token) <= 2:
        return [token]
    grams = [token[index : index + 2] for index in range(0, len(token) - 1)]
    if len(token) <= 6:
        return [token, *grams]
    return grams


def _support_term_weight(term: str) -> float:
    """Return deterministic local support term weight."""

    if _SUPPORT_NUMBER_RE.match(term):
        return 2.4
    if any(char.isdigit() for char in term):
        return 2.2
    if _SUPPORT_CJK_RE.match(term):
        return 1.5
    if len(term) >= 6:
        return 1.6
    return 1.0


def _claim_support(
    text: str,
    source_indexes: list[int],
    sources: list[JsonDict],
    raw_sources_by_url: dict[str, JsonDict],
) -> JsonDict:
    """Return a deterministic support score for one claim-source mapping."""

    citation_support = _claim_citation_support(text, source_indexes, sources, raw_sources_by_url)
    if citation_support["level"] == "unsupported":
        return {"score": 0, "level": "unsupported", "label": "引用未支撑", "citation_support": citation_support}
    if len(source_indexes) >= 2:
        return {"score": SUPPORT_SCORE_STRONG, "level": "strong", "label": "多来源支撑", "citation_support": citation_support}
    if len(source_indexes) == 1:
        return {"score": SUPPORT_SCORE_LINKED, "level": "linked", "label": "单来源关联", "citation_support": citation_support}
    return {"score": 0, "level": "unsupported", "label": "待补来源", "citation_support": citation_support}


def _claim_semantic_verification(source_indexes: list[int], support: JsonDict) -> JsonDict:
    """Return a conservative semantic verification precheck for one claim."""

    if source_indexes and support.get("level") == "unsupported":
        return {
            "score": 0,
            "level": "needs_evidence",
            "label": "引用未支撑",
            "note": "已标注来源，但本地引用支撑预检未发现正文主张与来源内容的足够重合；请补充来源或改写为假设，不代表事实真伪或语义蕴含。",
        }
    if len(source_indexes) >= 2:
        return {
            "score": SEMANTIC_SCORE_READY,
            "level": "ready_for_review",
            "label": "可进入语义复核",
            "note": "多来源映射足以进入人工语义复核；不代表事实真伪或语义蕴含。",
        }
    if len(source_indexes) == 1:
        return {
            "score": SEMANTIC_SCORE_LINKED,
            "level": "needs_review",
            "label": "单来源待语义复核",
            "note": "仅有单一来源映射；引用前需打开来源逐句比对，不代表事实真伪或语义蕴含。",
        }
    return {
        "score": 0,
        "level": "needs_evidence",
        "label": "缺证据",
        "note": "缺少可映射来源；请补充来源或改写为假设，不代表事实真伪或语义蕴含。",
    }


def _support_summary(claims: list[JsonDict]) -> JsonDict:
    """Return a user-facing source support summary for all public claims."""

    if not claims:
        return {
            "score": 0,
            "level": "missing",
            "label": "缺少来源支撑",
            "note": "未找到可评分的正文主张；该评分仅表示来源映射强弱，不代表事实真伪。",
        }
    score = round(sum(int(item.get("support_score") or 0) for item in claims) / len(claims))
    unsupported = sum(1 for item in claims if item.get("support_level") == "unsupported")
    if score <= 0 and unsupported:
        level = "missing"
        label = "缺少来源支撑"
        note = "正文主张缺少来源或引用未通过支撑预检；该评分仅表示来源映射强弱，不代表事实真伪。"
    elif score <= 0:
        level = "missing"
        label = "缺少来源支撑"
        note = "正文主张尚未关联可核验来源；该评分仅表示来源映射强弱，不代表事实真伪。"
    elif unsupported or score < 80:
        level = "partial"
        label = "部分主张支撑不足"
        note = f"来源支撑评分 {score}/100；{unsupported} 条主张仍需补充来源；仅表示来源映射强弱，不代表事实真伪。"
    else:
        level = "strong"
        label = "来源支撑较强"
        note = f"来源支撑评分 {score}/100；所有抽取主张均已关联来源；仅表示来源映射强弱，不代表事实真伪。"
    return {"score": score, "level": level, "label": label, "note": note}


def _review_actions(claims: list[JsonDict]) -> JsonDict:
    """Return user-facing next steps for claims that still need source review."""

    if not claims:
        return {
            "level": "missing",
            "label": "先生成可核验主张",
            "note": "未找到可执行的主张核验行动；请先生成带来源的研究答案。",
            "claim_ids": [],
            "items": [],
        }
    unsupported = [
        item
        for item in claims
        if item.get("status") == "missing" or item.get("support_level") == "unsupported"
    ]
    if not unsupported:
        return {
            "level": "ready",
            "label": "进入来源核验",
            "note": "所有抽取主张已映射来源；请打开来源逐条核对。",
            "claim_ids": [],
            "items": [],
        }
    items = [
        {
            "claim_id": str(item.get("id") or ""),
            "action": "补充来源或改写为假设",
            "text": _safe_text(item.get("text"), REVIEW_ACTION_TEXT_LIMIT),
        }
        for item in unsupported
        if item.get("id")
    ]
    claim_ids = [item["claim_id"] for item in items]
    return {
        "level": "needs_sources",
        "label": "待补来源行动",
        "note": f"{len(items)} 条主张需要补充来源或改写为假设。",
        "claim_ids": claim_ids,
        "items": items,
    }


def _quality_gate(claims: list[JsonDict], support_summary: JsonDict, review_actions: JsonDict) -> JsonDict:
    """Return a user-facing source-mapping readiness gate."""

    score = int(support_summary.get("score") or 0)
    claim_ids = [str(item) for item in review_actions.get("claim_ids") or [] if str(item)]
    if not claims:
        return {
            "level": "insufficient",
            "label": "证据不足",
            "score": 0,
            "action": "generate_with_sources",
            "claim_ids": [],
            "verification": "source_mapping_only",
            "note": "未找到可核验主张；请先生成带来源的研究答案。该门禁仅检查来源映射，不代表事实真伪。",
        }
    if review_actions.get("level") == "needs_sources":
        count = len(claim_ids)
        return {
            "level": "review_required",
            "label": "引用前需复核",
            "score": score,
            "action": "add_sources_or_rewrite",
            "claim_ids": claim_ids,
            "verification": "source_mapping_only",
            "note": _safe_text(
                f"{count} 条主张缺少来源或支撑不足；引用前请补充来源或改写为假设。该门禁仅检查来源映射，不代表事实真伪。",
                QUALITY_GATE_NOTE_LIMIT,
            ),
        }
    if support_summary.get("level") == "strong":
        return {
            "level": "ready",
            "label": "可进入引用",
            "score": score,
            "action": "open_sources_to_verify",
            "claim_ids": [],
            "verification": "source_mapping_only",
            "note": "所有抽取主张已映射来源；引用前仍建议打开来源核对关键事实。该门禁仅检查来源映射，不代表事实真伪。",
        }
    return {
        "level": "review_required",
        "label": "引用前需复核",
        "score": score,
        "action": "open_sources_to_verify",
        "claim_ids": [],
        "verification": "source_mapping_only",
        "note": "来源映射支撑偏弱；引用前请打开来源逐条核对。该门禁仅检查来源映射，不代表事实真伪。",
    }


def _semantic_verification(claims: list[JsonDict]) -> JsonDict:
    """Return a public semantic verification precheck derived from source alignment."""

    caveat = "启发式来源一致性预检，基于引用映射、来源数量与引用支撑信号；不代表事实真伪或语义蕴含。"
    if not claims:
        return {
            "level": "needs_evidence",
            "label": "缺少语义核验输入",
            "score": 0,
            "action": "generate_with_sources",
            "mode": "heuristic_source_alignment",
            "claim_ids": [],
            "note": f"未找到可进入语义核验的正文主张。{caveat}",
        }

    score = _round_half_up(sum(int(item.get("semantic_score") or 0) for item in claims) / len(claims))
    claim_ids = [
        str(item.get("id") or "")
        for item in claims
        if str(item.get("id") or "") and item.get("semantic_level") != "ready_for_review"
    ]
    needs_evidence = any(item.get("semantic_level") == "needs_evidence" for item in claims)
    citation_mismatch = any(
        item.get("semantic_level") == "needs_evidence" and item.get("status") == "mapped" and item.get("support_level") == "unsupported"
        for item in claims
    )
    if needs_evidence:
        level = "needs_evidence"
        label = "需补证据"
        action = "add_sources_or_mark_assumption"
        reason = "，其中包含引用未支撑主张" if citation_mismatch else ""
        detail = f"{len(claim_ids)} 条主张仍需补证据或语义复核{reason}。"
    elif claim_ids:
        level = "needs_review"
        label = "待语义核验"
        action = "open_sources_to_compare"
        detail = f"{len(claim_ids)} 条单来源主张需打开来源逐句比对。"
    else:
        level = "ready_for_review"
        label = "可进入语义复核"
        action = "open_sources_to_compare"
        detail = "多来源映射已就绪，请打开来源逐条做语义比对。"
    return {
        "level": level,
        "label": label,
        "score": score,
        "action": action,
        "mode": "heuristic_source_alignment",
        "claim_ids": claim_ids,
        "note": _safe_text(f"{detail}{caveat}", SEMANTIC_VERIFICATION_NOTE_LIMIT),
    }


def _claims_with_provider_verification(claims: list[JsonDict], provider_verification: Any) -> tuple[JsonDict, list[JsonDict]]:
    """Return provider summary and claims enriched with sanitized provider verdicts."""

    known_claim_ids = {str(claim.get("id") or "") for claim in claims}
    provider_items = _provider_items(provider_verification, known_claim_ids)
    seen_provider_ids = {item["claim_id"] for item in provider_items}
    for claim in claims:
        item = _provider_item_from_claim(claim, known_claim_ids)
        if item and item["claim_id"] not in seen_provider_ids:
            provider_items.append(item)
            seen_provider_ids.add(item["claim_id"])
    item_by_claim_id = {item["claim_id"]: item for item in provider_items}
    enabled = _provider_enabled(provider_verification)
    enriched: list[JsonDict] = []
    for claim in claims:
        claim_id = str(claim.get("id") or "")
        item = item_by_claim_id.get(claim_id)
        if item:
            provider_fields = _provider_claim_fields(item)
        elif enabled:
            provider_fields = {
                "provider_verdict": "not_checked",
                "provider_score": 0,
                "provider_label": "模型未返回结果",
                "provider_note": "已启用模型语义核验，但 provider 未返回该主张的结果。",
            }
        else:
            provider_fields = _default_provider_claim_fields()
        enriched.append({**claim, **provider_fields})
    return _provider_summary(provider_verification, provider_items), enriched


def _provider_summary(provider_verification: Any, items: list[JsonDict]) -> JsonDict:
    """Return a top-level provider-backed semantic verification summary."""

    enabled = _provider_enabled(provider_verification)
    status = _provider_status(provider_verification, enabled=enabled)
    supported = sum(1 for item in items if item.get("verdict") == "supported")
    contradicted = sum(1 for item in items if item.get("verdict") == "contradicted")
    uncertain = sum(1 for item in items if item.get("verdict") == "uncertain")
    unsupported = sum(1 for item in items if item.get("verdict") == "unsupported")
    claim_ids = [item["claim_id"] for item in items if item.get("verdict") != "supported"]
    return {
        "enabled": enabled,
        "status": status,
        "mode": "provider_semantic_entailment",
        "model": _safe_text(_provider_value(provider_verification, "model"), 120),
        "checked_claim_count": len(items),
        "supported_claim_count": supported,
        "contradicted_claim_count": contradicted,
        "uncertain_claim_count": uncertain,
        "unsupported_claim_count": unsupported,
        "claim_ids": claim_ids,
        "note": _provider_note(provider_verification, status=status, checked_count=len(items)),
    }


def _provider_actions(claims: list[JsonDict], provider_summary: JsonDict) -> JsonDict:
    """Return user-facing next steps from provider verdicts."""

    if not provider_summary.get("enabled"):
        return {
            "level": "skipped",
            "label": "未启用模型核验",
            "note": "默认离线；启用模型语义核验后才生成模型核验行动。",
            "claim_ids": [],
            "items": [],
        }
    items = [_provider_action_item(item) for item in claims]
    items = [item for item in items if item]
    if not items:
        status = str(provider_summary.get("status") or "")
        if status in {"failed", "no_input", "no_results"}:
            return {
                "level": "unavailable",
                "label": "模型核验未完成",
                "note": "模型核验没有返回可执行行动；请保留来源映射和语义预检结果人工核对。",
                "claim_ids": [],
                "items": [],
            }
        return {
            "level": "ready",
            "label": "模型核验通过",
            "note": "模型未标出冲突、缺支撑或不确定主张；引用前仍建议打开来源抽查。",
            "claim_ids": [],
            "items": [],
        }
    verdicts = {str(item.get("verdict") or "") for item in items}
    if "contradicted" in verdicts:
        level = "conflict"
        label = "优先核对冲突"
    elif "unsupported" in verdicts:
        level = "needs_sources"
        label = "补充证据或改写"
    else:
        level = "needs_review"
        label = "人工复核"
    return {
        "level": level,
        "label": label,
        "note": f"{len(items)} 条主张需要根据模型核验结果继续处理。",
        "claim_ids": [str(item["claim_id"]) for item in items],
        "items": items,
    }


def _provider_action_item(claim: JsonDict) -> JsonDict | None:
    """Return one action item for an actionable provider verdict."""

    claim_id = _safe_text(claim.get("id"), 24)
    verdict = str(claim.get("provider_verdict") or "").strip().lower()
    action = PROVIDER_ACTIONS.get(verdict)
    if not claim_id or not action:
        return None
    return {
        "claim_id": claim_id,
        "verdict": verdict,
        "action": action,
        "text": _safe_text(claim.get("text"), PROVIDER_ACTION_TEXT_LIMIT),
    }


def _conflict_gate(
    *,
    claims: list[JsonDict],
    source_relevance: JsonDict,
    provider_summary: JsonDict,
    provider_actions: JsonDict,
) -> JsonDict:
    """Return the product-facing conflict gate for generated claims."""

    items: list[JsonDict] = []
    for action in provider_actions.get("items") or []:
        if not isinstance(action, dict):
            continue
        verdict = str(action.get("verdict") or "").strip().lower()
        if verdict == "contradicted":
            items.append(
                _conflict_gate_item(
                    kind="provider_contradiction",
                    claim_id=action.get("claim_id"),
                    severity=1,
                    action=action.get("action") or "核对冲突来源",
                    text=action.get("text"),
                )
            )
        elif verdict == "unsupported":
            items.append(
                _conflict_gate_item(
                    kind="provider_unsupported",
                    claim_id=action.get("claim_id"),
                    severity=2,
                    action=action.get("action") or "补充来源或改写为假设",
                    text=action.get("text"),
                )
            )
        elif verdict == "uncertain":
            items.append(
                _conflict_gate_item(
                    kind="provider_uncertain",
                    claim_id=action.get("claim_id"),
                    severity=3,
                    action=action.get("action") or "人工复核",
                    text=action.get("text"),
                )
            )

    source_level = str(source_relevance.get("level") or "")
    if source_level in {"missing", "unrelated", "weak"}:
        items.append(
            _conflict_gate_item(
                kind="source_mismatch",
                claim_id="",
                severity=2 if source_level in {"missing", "unrelated"} else 3,
                action="重新检索或补充更相关来源",
                text=source_relevance.get("note") or "来源与问题相关性不足，不能直接当作可靠证据。",
            )
        )

    items = sorted(items, key=lambda item: (int(item.get("severity") or 9), str(item.get("claim_id") or ""), str(item.get("kind") or "")))[:DEFAULT_CLAIM_LIMIT]
    claim_ids = [item["claim_id"] for item in items if item.get("claim_id")]
    provider_status = str(provider_summary.get("status") or "")
    provider_unavailable = (not provider_summary.get("enabled")) or provider_status in {"failed", "no_input", "no_results", "skipped"}
    if any(item.get("severity") == 1 for item in items):
        level = "conflict"
        label = "存在冲突"
        action = "resolve_conflicts_before_using"
        detail = "已有模型核验信号提示主张与来源可能冲突，引用前请先核对冲突来源。"
    elif items:
        level = "caution"
        label = "建议先核对"
        action = "review_before_using"
        detail = "未发现确定冲突，但存在缺支撑、未确认或来源相关性不足的信号。"
    elif not claims:
        level = "unavailable"
        label = "冲突核验不可用"
        action = "generate_with_sources"
        detail = "未找到可进入冲突核验的公开主张，请先生成带来源的研究答案。"
    elif provider_unavailable:
        level = "unavailable"
        label = "冲突未核验"
        action = "open_sources_to_verify"
        detail = "默认离线或模型核验未完成；当前未发现确定冲突，但也不能证明没有冲突。"
    else:
        level = "clear"
        label = "未发现冲突"
        action = "open_sources_to_verify"
        detail = "模型未标出冲突、缺支撑或不确定主张；关键事实仍建议打开来源抽查。"
    return {
        "level": level,
        "label": label,
        "action": action,
        "claim_ids": claim_ids,
        "note": _safe_text(f"{detail}冲突门禁只合成公开证据、来源相关性和可选模型核验信号，不代表事实真伪。", QUALITY_GATE_NOTE_LIMIT),
        "items": items,
    }


def _conflict_gate_item(*, kind: str, claim_id: Any, severity: int, action: Any, text: Any) -> JsonDict:
    """Return one sanitized conflict gate item."""

    return {
        "kind": _safe_text(kind, 48),
        "claim_id": _safe_text(claim_id, 24),
        "severity": max(1, min(9, int(severity))),
        "action": _safe_text(action, REVIEW_ACTION_TEXT_LIMIT),
        "text": _safe_text(text, CONFLICT_GATE_TEXT_LIMIT),
    }


def _evidence_repair(
    *,
    claims: list[JsonDict],
    quality_gate: JsonDict,
    source_relevance: JsonDict,
    semantic_verification: JsonDict,
    provider_actions: JsonDict,
    review_actions: JsonDict,
) -> JsonDict:
    """Return one product-facing evidence repair plan from provenance signals."""

    items: list[JsonDict] = []
    source_level = str(source_relevance.get("level") or "")
    if source_level in {"missing", "unrelated", "weak"}:
        items.append(
            _evidence_repair_item(
                kind="source_relevance",
                claim_id="",
                priority=1,
                action="重新检索或补充更相关来源",
                text=source_relevance.get("note") or "来源与问题相关性不足，请先补充更贴近问题的来源。",
            )
        )

    for item in provider_actions.get("items") or []:
        if not isinstance(item, dict):
            continue
        verdict = str(item.get("verdict") or "")
        priority = 1 if verdict == "contradicted" else 2 if verdict == "unsupported" else 3
        items.append(
            _evidence_repair_item(
                kind="provider_action",
                claim_id=item.get("claim_id"),
                priority=priority,
                action=item.get("action") or "人工复核",
                text=item.get("text"),
            )
        )

    existing = {(item["kind"], item["claim_id"]) for item in items}
    claims_by_id = {str(item.get("id") or ""): item for item in claims if str(item.get("id") or "")}
    for item in review_actions.get("items") or []:
        if not isinstance(item, dict):
            continue
        claim_id = _safe_text(item.get("claim_id"), 24)
        claim = claims_by_id.get(claim_id, {})
        is_citation_mismatch = _is_citation_support_unsupported(claim)
        kind = "citation_support" if is_citation_mismatch else "review_action"
        key = (kind, claim_id)
        if key in existing or any(existing_item["claim_id"] == claim_id for existing_item in items if claim_id):
            continue
        items.append(
            _evidence_repair_item(
                kind=kind,
                claim_id=claim_id,
                priority=2 if is_citation_mismatch else 3,
                action=(
                    "补充能支撑该主张的来源或改写为假设"
                    if is_citation_mismatch
                    else item.get("action") or "补充来源或改写为假设"
                ),
                text=item.get("text"),
            )
        )
        existing.add(key)

    semantic_claim_ids = [str(item) for item in semantic_verification.get("claim_ids") or [] if str(item)]
    review_claim_ids = {str(item) for item in review_actions.get("claim_ids") or [] if str(item)}
    for claim_id in semantic_claim_ids:
        if claim_id in review_claim_ids:
            continue
        if any(item["claim_id"] == claim_id for item in items):
            continue
        claim = next((item for item in claims if item.get("id") == claim_id), {})
        if str(claim.get("provider_verdict") or "").lower() == "supported":
            continue
        items.append(
            _evidence_repair_item(
                kind="semantic_review",
                claim_id=claim_id,
                priority=4,
                action="打开来源逐句比对",
                text=claim.get("text") or semantic_verification.get("note"),
            )
        )

    items = sorted(items, key=lambda item: (int(item.get("priority") or 9), str(item.get("claim_id") or ""), str(item.get("kind") or "")))[:DEFAULT_CLAIM_LIMIT]
    claim_ids = [item["claim_id"] for item in items if item.get("claim_id")]
    if any(item.get("kind") == "provider_action" and item.get("priority") == 1 for item in items):
        level = "conflict"
        label = "先处理冲突"
        action = "resolve_conflicts_before_using"
        detail = "模型核验或来源映射提示存在冲突，引用前请先核对冲突来源。"
    elif items:
        level = "needs_repair"
        label = "建议修复证据"
        action = "repair_evidence_before_using"
        detail = "部分来源或主张仍需补证据、改写为假设或人工复核。"
    elif not claims or quality_gate.get("level") == "insufficient":
        level = "insufficient"
        label = "先生成可核验证据"
        action = "generate_with_sources"
        detail = "未找到足够的可核验主张，请先生成带来源的研究答案。"
    else:
        level = "ready"
        label = "暂无证据修复项"
        action = "open_sources_to_verify"
        detail = "当前未发现必须先修复的证据项，引用前仍建议打开来源抽查。"
    return {
        "level": level,
        "label": label,
        "action": action,
        "claim_ids": claim_ids,
        "note": _safe_text(f"{detail}证据修复建议由来源相关性、引用映射、语义预检和可选模型核验合成；不代表事实真伪。", QUALITY_GATE_NOTE_LIMIT),
        "items": items,
    }


def _evidence_repair_item(*, kind: str, claim_id: Any, priority: int, action: Any, text: Any) -> JsonDict:
    """Return one sanitized evidence repair item."""

    return {
        "kind": _safe_text(kind, 40),
        "claim_id": _safe_text(claim_id, 24),
        "priority": max(1, min(9, int(priority))),
        "action": _safe_text(action, REVIEW_ACTION_TEXT_LIMIT),
        "text": _safe_text(text, EVIDENCE_REPAIR_TEXT_LIMIT),
    }


def _is_citation_support_unsupported(claim: JsonDict) -> bool:
    """Return whether a mapped claim cites a source that does not support it."""

    citation_support = claim.get("citation_support") if isinstance(claim.get("citation_support"), dict) else {}
    return (
        str(claim.get("status") or "") == "mapped"
        and str(claim.get("support_level") or "") == "unsupported"
        and str(citation_support.get("level") or "") == "unsupported"
    )


def _provider_items(provider_verification: Any, known_claim_ids: set[str]) -> list[JsonDict]:
    """Return sanitized provider result items for known claims."""

    if not isinstance(provider_verification, dict):
        return []
    raw_items = provider_verification.get("items")
    if not isinstance(raw_items, list):
        raw_items = provider_verification.get("claims") if isinstance(provider_verification.get("claims"), list) else []
    items: list[JsonDict] = []
    seen: set[str] = set()
    for item in raw_items:
        if not isinstance(item, dict) or len(items) >= DEFAULT_CLAIM_LIMIT:
            continue
        claim_id = _safe_text(item.get("claim_id") or item.get("claimId") or item.get("id"), 24)
        if claim_id not in known_claim_ids or claim_id in seen:
            continue
        seen.add(claim_id)
        verdict = _provider_verdict(item.get("verdict"))
        items.append(
            {
                "claim_id": claim_id,
                "verdict": verdict,
                "score": _provider_score(item.get("score")),
                "note": _provider_public_note(verdict),
            }
        )
    return items


def _provider_item_from_claim(claim: JsonDict, known_claim_ids: set[str]) -> JsonDict | None:
    """Return a sanitized provider item from already-normalized claim fields."""

    claim_id = _safe_text(claim.get("id"), 24)
    verdict = str(claim.get("provider_verdict") or claim.get("providerVerdict") or "").strip().lower()
    if claim_id not in known_claim_ids or verdict not in (PROVIDER_VERDICTS - {"not_checked"}):
        return None
    return {
        "claim_id": claim_id,
        "verdict": verdict,
        "score": _provider_score(claim.get("provider_score", claim.get("providerScore"))),
        "note": _provider_public_note(verdict),
    }


def _existing_provider_claim_fields(item: JsonDict) -> JsonDict:
    """Return provider fields already present on a public claim payload."""

    verdict = str(item.get("provider_verdict") or item.get("providerVerdict") or "").strip().lower()
    if verdict not in (PROVIDER_VERDICTS - {"not_checked"}):
        return {}
    return {
        "provider_verdict": verdict,
        "provider_score": _provider_score(item.get("provider_score", item.get("providerScore"))),
        "provider_label": PROVIDER_VERDICT_LABELS[verdict],
        "provider_note": _provider_public_note(verdict),
    }


def _provider_claim_fields(item: JsonDict) -> JsonDict:
    """Return provider fields for one claim item."""

    verdict = _provider_verdict(item.get("verdict"))
    return {
        "provider_verdict": verdict,
        "provider_score": _provider_score(item.get("score")),
        "provider_label": PROVIDER_VERDICT_LABELS[verdict],
        "provider_note": _provider_public_note(verdict),
    }


def _default_provider_claim_fields() -> JsonDict:
    """Return provider claim fields for the default offline path."""

    return {
        "provider_verdict": "not_checked",
        "provider_score": 0,
        "provider_label": PROVIDER_VERDICT_LABELS["not_checked"],
        "provider_note": "默认离线；未执行 provider-backed 模型语义核验。",
    }


def _provider_enabled(provider_verification: Any) -> bool:
    """Return whether a provider verification payload represents an opt-in run."""

    if not isinstance(provider_verification, dict):
        return False
    if provider_verification.get("enabled") is True:
        return True
    status = str(provider_verification.get("status") or "").strip().lower()
    return status in {"completed", "failed", "no_input", "no_results"}


def _provider_status(provider_verification: Any, *, enabled: bool) -> str:
    """Return a bounded provider verification status."""

    if not enabled:
        return "skipped"
    raw = str(_provider_value(provider_verification, "status") or "").strip().lower()
    return raw if raw in {"completed", "failed", "no_input", "no_results"} else "completed"


def _provider_note(provider_verification: Any, *, status: str, checked_count: int) -> str:
    """Return a public note for provider-backed semantic verification."""

    raw_note = _safe_text(_provider_value(provider_verification, "note"), PROVIDER_VERIFICATION_NOTE_LIMIT)
    if raw_note:
        return raw_note
    if status == "skipped":
        return "默认离线；设置 DEEPRESEARCH_PROVIDER_SEMANTIC_VERIFICATION=1 后才会使用本机 BYOK 模型做语义核验。"
    if status == "failed":
        return "模型语义核验未完成；请保留启发式预检结果并人工打开来源核对。"
    if status == "no_input":
        return "未找到可交给模型核验的公开主张。"
    if not checked_count:
        return "模型语义核验已启用，但 provider 未返回可用主张结果。"
    return "模型语义核验已基于可见主张和来源摘要执行；结果仍需人工核对。"


def _provider_verdict(value: Any) -> str:
    """Return a supported provider verdict value."""

    verdict = str(value or "").strip().lower()
    return verdict if verdict in PROVIDER_VERDICTS and verdict != "not_checked" else "uncertain"


def _provider_public_note(verdict: str) -> str:
    """Return a deterministic public note for provider verdicts."""

    return PROVIDER_VERDICT_NOTES.get(verdict, "模型语义核验未返回说明；结果仍需人工核对。")


def _provider_score(value: Any) -> int:
    """Return a bounded provider verification score."""

    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 0
    return max(0, min(100, score))


def _provider_value(provider_verification: Any, key: str) -> Any:
    """Return a field from a provider verification payload."""

    if not isinstance(provider_verification, dict):
        return ""
    return provider_verification.get(key)


def _raw_sources_by_safe_url(raw_sources: list[dict[str, Any]]) -> dict[str, JsonDict]:
    """Return raw source records keyed by their sanitized public URL."""

    by_url: dict[str, JsonDict] = {}
    for item in raw_sources:
        if not isinstance(item, dict):
            continue
        safe = sanitize_report_sources([item])
        if safe:
            by_url.setdefault(str(safe[0].get("url") or ""), item)
    return by_url


def _round_half_up(value: float) -> int:
    """Round a non-negative score using conventional half-up semantics."""

    return int(max(0.0, value) + 0.5)


def _safe_limit(value: Any) -> int:
    """Return a bounded claim count limit."""

    try:
        limit = int(value)
    except (TypeError, ValueError):
        limit = DEFAULT_CLAIM_LIMIT
    return min(max(limit, 0), DEFAULT_CLAIM_LIMIT)


def _safe_text(value: Any, limit: int) -> str:
    """Return compact public text without secrets or local paths."""

    text = _LOCAL_PATH_RE.sub("[local path]", redact_text(str(value or "")))
    text = " ".join(text.split()).strip()
    return text[: max(0, int(limit))]
