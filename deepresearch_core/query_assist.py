from __future__ import annotations

import re
from typing import Any


JsonDict = dict[str, Any]


_CHATGPT_SPACING_RE = re.compile(r"(?i)\bchat\s+gpt\b")
_GPT_SPACING_RE = re.compile(r"(?i)\bgpt\s+(\d+(?:\.\d+)?)\b")
_NIO_TYPO_RE = re.compile(r"蔚蓝(?=汽车|车|ES|ET|EC|EP|NIO|20|\s|$)")
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_VERSION_OR_AI_PRODUCT_RE = re.compile(
    r"(?i)\b(?:chatgpt|gpt|gemini|claude|qwen|kimi|openai|react|windows|python|npm)\b|\b\d+(?:\.\d+)?\b"
)
_VEHICLE_ENTITY_ALIASES = (
    (re.compile(r"蔚来(?:汽车|车)?|蔚蓝(?:汽车|车)?"), "NIO", "NIO / 蔚来汽车"),
    (re.compile(r"比亚迪"), "BYD", "BYD / 比亚迪"),
    (re.compile(r"小米\s*SU\s*7|小米SU7|小米汽车|小米EV|小米电动"), "Xiaomi SU7", "Xiaomi SU7 / 小米汽车"),
    (re.compile(r"理想汽车|理想L\d+|理想MEGA", re.IGNORECASE), "Li Auto", "Li Auto / 理想汽车"),
    (re.compile(r"小鹏汽车|小鹏(?:P|G)\d+|XPeng", re.IGNORECASE), "XPeng", "XPeng / 小鹏汽车"),
)
_VEHICLE_OFFICIAL_SOURCE_SITES = (
    (re.compile(r"蔚来(?:汽车|车)?|蔚蓝(?:汽车|车)?|\bNIO\b", re.IGNORECASE), "NIO", "ir.nio.com", "NIO investor relations"),
    (re.compile(r"理想汽车|理想L\d+|理想MEGA|\bLi Auto\b", re.IGNORECASE), "Li Auto", "ir.lixiang.com", "Li Auto investor relations"),
    (re.compile(r"小鹏汽车|小鹏(?:P|G)\d+|XPeng", re.IGNORECASE), "XPeng", "ir.xiaopeng.com", "XPeng investor relations"),
    (re.compile(r"小米\s*SU\s*7|小米SU7|小米汽车|小米EV|小米电动", re.IGNORECASE), "Xiaomi SU7", "ir.mi.com", "Xiaomi investor relations"),
)


def expand_query_items(items: list[JsonDict]) -> list[JsonDict]:
    """Return original query items plus conservative assisted variants."""

    expanded: list[JsonDict] = []
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            continue
        original = dict(item)
        original.setdefault("origin_branch", index)
        expanded.append(original)
        query = str(item.get("query") or "")
        assists = query_assist_variants(query)
        for assist in assists:
            assisted = dict(item)
            assisted["query"] = assist["query"]
            assisted["query_assist"] = assist
            assisted["origin_branch"] = original.get("origin_branch") or index
            if not assisted.get("dimension"):
                assisted["dimension"] = "辅助检索"
            expanded.append(assisted)
        source_base = assists[0]["query"] if assists else query
        for variant in source_seeking_variants(source_base):
            source_item = dict(item)
            source_item["query"] = variant["query"]
            source_item["query_strategy"] = variant
            source_item["origin_branch"] = original.get("origin_branch") or index
            if not source_item.get("dimension"):
                source_item["dimension"] = "来源查证"
            expanded.append(source_item)
    return expanded


def source_seeking_variants(query: Any) -> list[JsonDict]:
    """Return backup source-first searches for freshness-sensitive paths."""

    text = _compact_query(query)
    if not text:
        return []
    lower = text.lower()
    variants: list[JsonDict] = []
    variants.extend(_official_site_variants(text, lower))
    if _looks_like_financial_or_business_query(text, lower):
        variants.extend(
            [
                {
                    "query": f"{text} 官方 财报 投资者关系 交付 销量",
                    "label": "官方财报与投资者关系",
                    "kind": "source_seeking",
                },
                {
                    "query": f"{text} investor relations earnings deliveries official",
                    "label": "official investor sources",
                    "kind": "source_seeking",
                },
            ]
        )
    elif _looks_like_release_or_product_status_query(text, lower):
        variants.extend(
            [
                {
                    "query": f"{text} 官方 来源 发布说明",
                    "label": "官方发布说明",
                    "kind": "source_seeking",
                },
                {
                    "query": f"{text} official source release notes documentation",
                    "label": "official source documentation",
                    "kind": "source_seeking",
                },
            ]
        )
    else:
        variants.extend(
            [
                {
                    "query": f"{text} 官方 来源 最新",
                    "label": "官方来源",
                    "kind": "source_seeking",
                },
                {
                    "query": f"{text} primary source official announcement",
                    "label": "primary source",
                    "kind": "source_seeking",
                },
            ]
        )
    return _unique_assists(variants, original=text)


def query_assist_variants(query: Any) -> list[JsonDict]:
    """Return low-risk query rewrites for obvious entity or product-name drift."""

    text = _compact_query(query)
    if not text:
        return []
    variants: list[JsonDict] = []
    nio_rewrite = _NIO_TYPO_RE.sub("蔚来", text)
    if nio_rewrite != text:
        variants.append(
            {
                "query": nio_rewrite,
                "label": "蔚来汽车",
                "kind": "likely_entity_typo",
                "reason": "检测到新能源汽车品牌的常见误写，补充规范实体检索。",
            }
        )
    variants.extend(_vehicle_entity_alias_variants(nio_rewrite))
    chatgpt_rewrite = _CHATGPT_SPACING_RE.sub("ChatGPT", text)
    chatgpt_rewrite = _GPT_SPACING_RE.sub(lambda match: f"GPT-{match.group(1)}", chatgpt_rewrite)
    if chatgpt_rewrite != text:
        variants.append(
            {
                "query": chatgpt_rewrite,
                "label": "ChatGPT / GPT",
                "kind": "canonical_product_name",
                "reason": "规范产品名空格与版本写法，避免搜索召回被词面拆分影响。",
            }
        )
    return _unique_assists(variants, original=text)


def preferred_relevance_query(original_query: Any, sources: list[JsonDict]) -> str:
    """Return the query that best reflects the sources actually retrieved."""

    original = _compact_query(original_query)
    for source in sources:
        if not isinstance(source, dict):
            continue
        assist = source.get("query_assist")
        if not isinstance(assist, dict):
            continue
        query = _compact_query(assist.get("query"))
        if query:
            return query
    for source in sources:
        if not isinstance(source, dict):
            continue
        strategy = source.get("query_strategy")
        if not isinstance(strategy, dict):
            continue
        query = _compact_query(strategy.get("query"))
        if query:
            return query
    for assist in query_assist_variants(original):
        if assist.get("kind") not in {"likely_entity_typo", "canonical_product_name"}:
            continue
        query = _compact_query(assist.get("query"))
        if query:
            return query
    return original


def query_assist_notice(sources: list[JsonDict]) -> str:
    """Return a short user-facing note when assisted queries contributed sources."""

    labels: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        assist = source.get("query_assist")
        if not isinstance(assist, dict):
            continue
        label = _compact_label(assist.get("label") or assist.get("query"))
        if label and label not in labels:
            labels.append(label)
    if not labels:
        return ""
    joined = "、".join(f"“{label}”" for label in labels[:3])
    return f"检索提示：已同时按{joined}补充检索，以降低实体误写或产品名拆分带来的漏召回。"


def _unique_assists(variants: list[JsonDict], *, original: str) -> list[JsonDict]:
    seen = {original.lower()}
    unique: list[JsonDict] = []
    for variant in variants:
        query = _compact_query(variant.get("query"))
        if not query or query.lower() in seen:
            continue
        seen.add(query.lower())
        copied = dict(variant)
        copied["query"] = query
        unique.append(copied)
    return unique


def _compact_query(value: Any) -> str:
    return " ".join(str(value or "").replace("\r", " ").replace("\n", " ").split()).strip()


def _compact_label(value: Any) -> str:
    return _compact_query(value)[:48].strip()


def _looks_like_financial_or_business_query(text: str, lower: str) -> bool:
    terms = (
        "业绩",
        "财报",
        "营收",
        "利润",
        "销量",
        "交付",
        "股价",
        "估值",
        "revenue",
        "earnings",
        "delivery",
        "deliveries",
        "stock",
        "valuation",
        "sales",
    )
    return any(term in text or term in lower for term in terms)


def _looks_like_release_or_product_status_query(text: str, lower: str) -> bool:
    terms = (
        "发布",
        "上线",
        "版本",
        "模型",
        "是否存在",
        "有没有",
        "新能力",
        "release",
        "released",
        "launch",
        "version",
        "model",
        "available",
        "availability",
        "status",
    )
    return any(term in text or term in lower for term in terms) or bool(_VERSION_OR_AI_PRODUCT_RE.search(lower))


def _vehicle_entity_alias_variants(text: str) -> list[JsonDict]:
    """Return official-source oriented English alias queries for known vehicle entities."""

    variants: list[JsonDict] = []
    for pattern, alias, label in _VEHICLE_ENTITY_ALIASES:
        if not pattern.search(text):
            continue
        query = _vehicle_entity_alias_query(text, alias)
        if not query:
            continue
        variants.append(
            {
                "query": query,
                "label": label,
                "kind": "official_entity_alias",
                "reason": "补充车企或车型英文官方名称检索，避免中文问题漏掉英文投资者关系、交付或公告来源。",
            }
        )
    return variants


def _official_site_variants(text: str, lower: str) -> list[JsonDict]:
    """Return official-site searches for known entities in freshness-sensitive questions."""

    if not (_looks_like_financial_or_business_query(text, lower) or _looks_like_release_or_product_status_query(text, lower)):
        return []
    years = _YEAR_RE.findall(text)
    year_part = " ".join(years[:2])
    variants: list[JsonDict] = []
    for pattern, alias, site, label in _VEHICLE_OFFICIAL_SOURCE_SITES:
        if not pattern.search(text):
            continue
        tail = (
            "financial results earnings deliveries sales investor relations"
            if _looks_like_financial_or_business_query(text, lower)
            else "official announcement release notes deliveries sales"
        )
        variants.append(
            {
                "query": _compact_query(f"site:{site} {alias} {year_part} {tail}"),
                "label": label,
                "kind": "source_seeking",
            }
        )
    return variants


def _vehicle_entity_alias_query(text: str, alias: str) -> str:
    """Return one compact official-source query for a vehicle entity alias."""

    years = _YEAR_RE.findall(text)
    year_part = " ".join(years[:2])
    lower = text.lower()
    if _looks_like_financial_or_business_query(text, lower):
        parts = [alias, year_part, "earnings financial results deliveries sales investor relations official"]
    elif _looks_like_release_or_product_status_query(text, lower):
        parts = [alias, year_part, "official announcement release notes deliveries sales"]
    else:
        parts = [alias, year_part, "official source"]
    return _compact_query(" ".join(part for part in parts if part))
