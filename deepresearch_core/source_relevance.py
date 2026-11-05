import re
from typing import Any, Iterable

from deepresearch_core.report_sources import sanitize_report_sources
from deepresearch_core.standalone.llm import redact_text


JsonDict = dict[str, Any]
SOURCE_RELEVANCE_NOTE_LIMIT = 240
SOURCE_RELEVANCE_ITEM_LIMIT = 12
SOURCE_RELEVANCE_TEXT_LIMIT = 4200
_TOKEN_RE = re.compile(r"(?i)[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*|\d+(?:\.\d+)?|[\u4e00-\u9fff]{2,}")
_CJK_RE = re.compile(r"^[\u4e00-\u9fff]+$")
_NUMBER_RE = re.compile(r"^\d+(?:\.\d+)?$")
_STOP_TERMS = frozenset(
    {
        "什么",
        "怎么",
        "如何",
        "是否",
        "有没有",
        "多少",
        "是谁",
        "当前",
        "现在",
        "最新",
        "最近",
        "这个",
        "那个",
        "一下",
        "请问",
        "帮我",
        "分析",
        "报告",
        "研究",
        "the",
        "a",
        "an",
        "and",
        "or",
        "is",
        "are",
        "what",
        "how",
        "who",
        "when",
        "latest",
        "current",
        "recent",
        "report",
        "research",
    }
)
_CRITICAL_SOURCE_TERMS = frozenset(
    {
        "ceo",
        "cfo",
        "cto",
        "president",
        "prime",
        "minister",
        "price",
        "stock",
        "rate",
        "weather",
        "release",
        "released",
        "launch",
        "launched",
        "status",
        "exist",
        "exists",
        "change",
        "changes",
        "update",
        "updates",
        "股价",
        "价格",
        "汇率",
        "利率",
        "天气",
        "发布",
        "上线",
        "进展",
        "变化",
        "更新",
        "执行",
        "现任",
        "首相",
        "总统",
        "法案",
        "政策",
        "监管",
    }
)
_ENTITY_ALIAS_WEIGHT = 2.4
_SEMANTIC_ALIAS_WEIGHT = 1.0
_QUERY_ENTITY_ALIASES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"蔚来(?:汽车|车)?|蔚蓝(?:汽车|车)?", re.IGNORECASE), ("nio",)),
    (re.compile(r"比亚迪", re.IGNORECASE), ("byd",)),
    (re.compile(r"小米\s*su\s*7|小米su7|小米汽车|小米ev|小米电动", re.IGNORECASE), ("xiaomi", "su7")),
    (re.compile(r"理想汽车|理想l\d+|理想mega", re.IGNORECASE), ("li", "auto")),
    (re.compile(r"小鹏汽车|小鹏(?:p|g)\d+|xpeng", re.IGNORECASE), ("xpeng",)),
)
_QUERY_SEMANTIC_ALIASES: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"业绩|财报|营收|收入|利润|交付|销量|earnings|revenue|deliver", re.IGNORECASE), ("financial", "results", "earnings", "revenue", "deliveries", "sales")),
    (re.compile(r"发布|上线|推出|版本|是否存在|有没有|release|available|availability|launch|version", re.IGNORECASE), ("release", "released", "available", "launch", "version")),
)


def build_source_relevance(query: Any, sources: Iterable[dict[str, Any]] | None) -> JsonDict:
    """Return a public query-to-source relevance precheck."""

    raw_sources = list(sources or [])
    safe_sources = sanitize_report_sources(raw_sources)
    query_terms = _weighted_query_terms(query)
    if not safe_sources:
        return _source_relevance_summary(
            level="missing",
            label="缺少来源",
            score=0,
            relevant_count=0,
            items=[],
        )
    if not query_terms:
        return _source_relevance_summary(
            level="unknown",
            label="未评估来源相关性",
            score=0,
            relevant_count=0,
            items=[_source_relevance_item(index, source, 0, "unknown", []) for index, source in enumerate(safe_sources, start=1)],
        )

    raw_by_url = _raw_sources_by_safe_url(raw_sources)
    items = []
    for index, source in enumerate(safe_sources[:SOURCE_RELEVANCE_ITEM_LIMIT], start=1):
        raw = raw_by_url.get(source["url"], source)
        score, matched_terms = _source_score(query_terms, _source_terms(raw, source))
        items.append(_source_relevance_item(index, source, score, _item_level(score), matched_terms))

    top_score = max((int(item["score"]) for item in items), default=0)
    relevant_count = sum(1 for item in items if item.get("level") == "relevant")
    if top_score >= 35 and relevant_count:
        level = "ready"
        label = "来源相关"
    elif top_score >= 18:
        level = "weak"
        label = "来源相关性偏弱"
    else:
        level = "unrelated"
        label = "来源相关性不足"
    return _source_relevance_summary(level=level, label=label, score=top_score, relevant_count=relevant_count, items=items)


def normalize_source_relevance(value: Any, *, query: Any = "", sources: Iterable[dict[str, Any]] | None = None) -> JsonDict:
    """Return safe source relevance, recomputing when a query is available."""

    if str(query or "").strip():
        return build_source_relevance(query, sources)
    safe_sources = sanitize_report_sources(list(sources or []))
    if isinstance(value, dict):
        raw_items = value.get("items") if isinstance(value.get("items"), list) else []
        safe_by_url = {source["url"]: (index, source) for index, source in enumerate(safe_sources, start=1)}
        items = []
        for item in raw_items[:SOURCE_RELEVANCE_ITEM_LIMIT]:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if safe_by_url and url not in safe_by_url:
                continue
            if safe_by_url:
                index, source = safe_by_url[url]
            else:
                safe_item = sanitize_report_sources([item])
                if not safe_item:
                    continue
                index, source = _bounded_index(item.get("index")), safe_item[0]
            score = _bounded_score(item.get("score"))
            items.append(_source_relevance_item(index, source, score, _item_level(score), _safe_matched_terms(item.get("matched_terms"))))
        if items or not safe_sources:
            score = _bounded_score(value.get("score"))
            level = _summary_level(value.get("level"), score=score)
            return _source_relevance_summary(
                level=level,
                label=_summary_label(level),
                score=score,
                relevant_count=sum(1 for item in items if item.get("level") == "relevant"),
                items=items,
            )
    return build_source_relevance("", safe_sources)


def has_relevant_source(query: Any, sources: Iterable[dict[str, Any]] | None) -> bool:
    """Return whether at least one source is sufficiently related to the query."""

    relevance = build_source_relevance(query, sources)
    return relevance.get("level") in {"ready", "weak"}


def _source_relevance_summary(*, level: str, label: str, score: int, relevant_count: int, items: list[JsonDict]) -> JsonDict:
    """Return the public source relevance summary envelope."""

    total = len(items)
    if level == "missing":
        detail = "本次运行没有可展示的 HTTP(S) 来源。"
        action = "configure_search_or_add_sources"
    elif level == "unknown":
        detail = "缺少可评估的问题文本，仅保留来源清单。"
        action = "ask_with_clear_query"
    elif level == "ready":
        detail = f"{relevant_count} 个来源与问题关键词匹配度较高。"
        action = "answer_with_sources"
    elif level == "weak":
        detail = "来源覆盖了部分关键词，但关键实体、版本或时间点仍需复核。"
        action = "refine_search_or_open_sources"
    else:
        detail = "未找到与问题关键实体、版本或时间点匹配的来源。"
        action = "refine_search_or_add_sources"
    note = _safe_note(f"{detail}该指标是查询-来源相关性预检，不代表事实真伪或语义蕴含。")
    return {
        "level": level,
        "label": label,
        "score": _bounded_score(score),
        "relevant_source_count": int(max(0, relevant_count)),
        "source_count": total,
        "action": action,
        "note": note,
        "items": items,
    }


def _source_relevance_item(index: int, source: JsonDict, score: int, level: str, matched_terms: list[str]) -> JsonDict:
    """Return one safe source relevance item."""

    return {
        "index": int(index),
        "title": _safe_label(source.get("title"), 180),
        "url": _safe_label(source.get("url"), 512),
        "site": _safe_label(source.get("site"), 120),
        "score": _bounded_score(score),
        "level": level,
        "matched_terms": _safe_matched_terms(matched_terms),
    }


def _raw_sources_by_safe_url(raw_sources: list[dict[str, Any]]) -> dict[str, JsonDict]:
    """Return raw source records keyed by their sanitized URL."""

    by_url: dict[str, JsonDict] = {}
    for item in raw_sources:
        safe = sanitize_report_sources([item])
        if safe:
            by_url.setdefault(safe[0]["url"], item)
    return by_url


def _weighted_query_terms(query: Any) -> dict[str, float]:
    """Return public query terms with deterministic weights."""

    raw_query = str(query or "")
    terms: dict[str, float] = {}
    for token in _tokens(raw_query):
        if token in _STOP_TERMS:
            continue
        terms[token] = max(terms.get(token, 0.0), _term_weight(token))
    for pattern, aliases in _QUERY_ENTITY_ALIASES:
        if not pattern.search(raw_query):
            continue
        for alias in aliases:
            terms[alias] = max(terms.get(alias, 0.0), _ENTITY_ALIAS_WEIGHT)
    for pattern, aliases in _QUERY_SEMANTIC_ALIASES:
        if not pattern.search(raw_query):
            continue
        for alias in aliases:
            terms[alias] = max(terms.get(alias, 0.0), _SEMANTIC_ALIAS_WEIGHT)
    return terms


def _source_terms(raw_source: JsonDict, safe_source: JsonDict) -> set[str]:
    """Return searchable terms from a source without exposing them."""

    parts = [
        safe_source.get("title", ""),
        safe_source.get("site", ""),
        safe_source.get("url", ""),
        raw_source.get("title", ""),
        raw_source.get("description", ""),
        raw_source.get("snippet", ""),
        raw_source.get("markdown", ""),
    ]
    return set(_tokens(" ".join(str(part or "") for part in parts)[:SOURCE_RELEVANCE_TEXT_LIMIT]))


def _source_score(query_terms: dict[str, float], source_terms: set[str]) -> tuple[int, list[str]]:
    """Return a bounded relevance score and matched terms."""

    if not query_terms:
        return 0, []
    matched = [term for term in query_terms if term in source_terms]
    total_weight = sum(query_terms.values())
    matched_weight = sum(query_terms[term] for term in matched)
    score = round((matched_weight / total_weight) * 100) if total_weight else 0
    numeric_terms = [term for term in query_terms if _NUMBER_RE.match(term) or any(char.isdigit() for char in term)]
    if numeric_terms and not any(term in matched for term in numeric_terms):
        score = min(score, 30)
    critical_terms = [term for term in query_terms if term in _CRITICAL_SOURCE_TERMS]
    if critical_terms and not any(term in matched for term in critical_terms):
        score = min(score, 30)
    entity_terms = [term for term in query_terms if term not in numeric_terms]
    if entity_terms and not any(term in matched for term in entity_terms):
        score = min(score, 25)
    return _bounded_score(score), sorted(matched, key=lambda term: (-query_terms[term], term))[:8]


def _tokens(text: str) -> list[str]:
    """Return compact lower-case lexical tokens for Chinese and ASCII text."""

    normalized = redact_text(str(text or "")).lower()
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(normalized):
        token = raw.strip("._-")
        if not token:
            continue
        if _CJK_RE.match(token):
            tokens.extend(_cjk_tokens(token))
            continue
        tokens.append(token)
        for part in re.split(r"[-_.]+", token):
            if part and part != token:
                tokens.append(part)
    return [token for token in tokens if token and token not in _STOP_TERMS]


def _cjk_tokens(token: str) -> list[str]:
    """Return CJK word-like tokens and bigrams."""

    if len(token) <= 2:
        return [token]
    grams = [token[index : index + 2] for index in range(0, len(token) - 1)]
    if len(token) <= 6:
        return [token, *grams]
    return grams


def _term_weight(term: str) -> float:
    """Return deterministic weight for a query term."""

    if _NUMBER_RE.match(term):
        return 2.4
    if any(char.isdigit() for char in term):
        return 2.2
    if _CJK_RE.match(term):
        return 1.5
    if len(term) >= 6:
        return 1.6
    return 1.0


def _item_level(score: int) -> str:
    """Return item-level source relevance label."""

    if score >= 35:
        return "relevant"
    if score >= 18:
        return "weak"
    return "unrelated"


def _summary_level(value: Any, *, score: int) -> str:
    """Return a safe summary level."""

    level = str(value or "").strip().lower()
    if level in {"missing", "unknown", "ready", "weak", "unrelated"}:
        return level
    if score >= 35:
        return "ready"
    if score >= 18:
        return "weak"
    return "unrelated"


def _summary_label(level: str) -> str:
    """Return the public label for a source relevance level."""

    return {
        "missing": "缺少来源",
        "unknown": "未评估来源相关性",
        "ready": "来源相关",
        "weak": "来源相关性偏弱",
        "unrelated": "来源相关性不足",
    }.get(level, "来源相关性不足")


def _safe_matched_terms(value: Any) -> list[str]:
    """Return safe matched query terms for public metadata."""

    if not isinstance(value, list):
        return []
    terms = []
    seen = set()
    for item in value:
        term = _safe_label(item, 48).lower()
        if not term or term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= 8:
            break
    return terms


def _bounded_score(value: Any) -> int:
    """Return a 0-100 integer score."""

    try:
        score = int(round(float(value)))
    except (TypeError, ValueError):
        score = 0
    return max(0, min(100, score))


def _bounded_index(value: Any) -> int:
    """Return a positive public source index."""

    try:
        index = int(value)
    except (TypeError, ValueError):
        index = 0
    return max(1, min(SOURCE_RELEVANCE_ITEM_LIMIT, index))


def _safe_note(value: Any) -> str:
    """Return compact public note text."""

    return _safe_label(value, SOURCE_RELEVANCE_NOTE_LIMIT)


def _safe_label(value: Any, limit: int) -> str:
    """Return compact safe public text."""

    text = redact_text(str(value or ""))
    text = re.sub(r"(?i)file:///[^\s,;)]+", "[local path]", text)
    text = re.sub(r"(?i)\b[A-Z]:\\[^\s,;)]+", "[local path]", text)
    text = re.sub(r"(?i)\b(?:/home|/users|/var|/tmp|/mnt)/[^\s,;)]+", "[local path]", text)
    text = " ".join(text.split()).strip()
    return text[: max(0, int(limit))]
