import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass(frozen=True)
class WorkflowInput:
    """Normalized workflow input."""

    user_query: str
    chat_history: List[Dict[str, str]] = field(default_factory=list)
    previous_search_context: str = ""
    messages: List[Dict[str, str]] = field(default_factory=list)
    research_mode: str = ""
    evidence_repair_request: Dict[str, Any] = field(default_factory=dict)
    research_plan_preview_only: bool = False
    research_plan_confirmation: Dict[str, Any] = field(default_factory=dict)
    research_plan_confirmation_provided: bool = False


@dataclass(frozen=True)
class RouteDecision:
    """Research route and effort settings."""

    level: str
    routing_decision: str
    effort_level: int
    label: str
    search_branches: int
    section_count: int
    enable_validation: bool
    enable_fact_check: bool
    requires_search: bool = False
    boundary_label: str = ""
    boundary_reason: str = ""
    user_notice: str = ""


PROBE_KEYWORDS = ("系统提示词", "system prompt", "内部规则", "prompt injection", "源码", "source code")
PUBLIC_RESEARCH_MODES = frozenset({"fast_search", "multiple_search", "deep_research", "models_council"})
LLM_PROVIDER_REQUIRED_FIELDS = frozenset({"LLM Base URL", "LLM API Key", "LLM Model"})
FRESHNESS_TERMS = (
    "2024",
    "2025",
    "2026",
    "今年",
    "本年",
    "去年",
    "明年",
    "本月",
    "上月",
    "下月",
    "本周",
    "上周",
    "下周",
    "昨天",
    "昨日",
    "明天",
    "明日",
    "截至",
    "截止",
    "至今",
    "销量",
    "最新",
    "今天",
    "最近",
    "当前",
    "现在",
    "发布",
    "推出",
    "上线",
    "更新",
    "版本",
    "型号",
    "价格",
    "新闻",
    "政策",
    "法规",
    "法案",
    "法律",
    "条例",
    "监管",
    "合规",
    "汇率",
    "股价",
    "行情",
    "利率",
    "市值",
    "排名",
    "榜单",
    "官网",
    "下载",
    "开源",
    "融资",
    "财报",
    "销量",
    "评价",
    "评测",
    "怎么样",
    "有哪些变化",
    "现任",
    "latest",
    "newest",
    "current",
    "today",
    "recent",
    "this year",
    "last year",
    "next year",
    "this month",
    "last month",
    "next month",
    "this week",
    "last week",
    "next week",
    "yesterday",
    "tomorrow",
    "as of",
    "so far",
    "released",
    "launch",
    "availability",
    "release date",
    "version",
    "price",
    "news",
    "policy",
    "law",
    "exchange rate",
    "stock price",
    "interest rate",
    "ranking",
    "download",
    "open source",
    "funding",
    "earnings",
    "review",
)
FRESHNESS_YEAR_FLOOR = 2024
YEAR_RE = re.compile(r"(?<!\d)(20\d{2})(?!\d)")
VOLATILE_ROLE_TERMS = (
    "ceo",
    "cto",
    "cfo",
    "coo",
    "founder",
    "president",
    "prime minister",
    "chairman",
    "负责人",
    "创始人",
    "董事长",
    "总裁",
    "总统",
    "首相",
    "部长",
    "市长",
)
HISTORICAL_IDENTITY_TERMS = (
    "第一任",
    "首任",
    "前任",
    "曾任",
    "历任",
    "创始人",
    "共同创始人",
    "发明",
    "创立",
    "创建",
    "成立",
    "建立",
    "出生",
    "去世",
    "who was ",
    "when was ",
    "when did ",
    "invented",
    "founded",
    "founder",
    "co-founder",
    "first president",
    "first ceo",
    "former ",
)
GENERIC_CONCEPT_PREFIXES = (
    "解释一下",
    "介绍一下",
    "怎么理解",
    "如何理解",
    "怎样理解",
    "什么是",
    "什么叫",
    "解释",
    "what is ",
    "explain ",
)
GENERIC_CONCEPT_SUFFIXES = (
    "是什么",
    "是啥",
    "什么意思",
    "的定义",
)
CONCEPT_FRESHNESS_BLOCK_TERMS = (
    "最新",
    "最近",
    "当前",
    "现在",
    "今天",
    "今年",
    "本月",
    "本周",
    "发布",
    "推出",
    "上线",
    "更新",
    "变化",
    "进展",
    "现任",
    "latest",
    "newest",
    "current",
    "today",
    "recent",
    "release",
    "released",
    "availability",
)
GENERIC_REGULATORY_CONCEPTS = (
    "法律",
    "法案",
    "法规",
    "政策",
    "公共政策",
    "监管",
    "合规",
    "条例",
)
REGULATORY_TARGET_TERMS = (
    "法律",
    "法案",
    "法规",
    "条例",
    "政策",
    "监管",
    "合规",
)
REGULATORY_ENTITY_TERMS = (
    "欧盟",
    "美国",
    "中国",
    "加州",
    "英国",
    "日本",
    "韩国",
    "新加坡",
    "ai",
    "人工智能",
    "数据",
    "隐私",
)
VOLATILE_METRIC_TARGET_TERMS = (
    "价格",
    "房价",
    "租金",
    "股价",
    "汇率",
    "利率",
    "市值",
    "估值",
    "市盈率",
    "销量",
    "销售额",
    "排名",
    "榜单",
    "财报",
    "融资",
    "行情",
    "人口",
    "gdp",
    "国内生产总值",
    "失业率",
    "通胀",
    "通货膨胀",
    "cpi",
    "ppi",
    "票房",
    "营收",
    "收入",
    "利润",
    "用户数",
    "月活",
    "日活",
    "市场份额",
    "市占率",
    "出货量",
    "装机量",
    "产量",
)
METRIC_ENTITY_TERMS = (
    "公司",
    "集团",
    "股份",
    "中国",
    "美国",
    "欧盟",
    "日本",
    "韩国",
    "英国",
    "德国",
    "法国",
    "印度",
    "新加坡",
    "北京",
    "上海",
    "深圳",
    "香港",
    "全球",
    "世界",
    "全国",
    "openai",
    "apple",
    "苹果",
    "tesla",
    "特斯拉",
    "小米",
    "比亚迪",
    "微软",
    "google",
    "谷歌",
    "美元",
    "人民币",
    "比特币",
    "ethereum",
    "以太坊",
    "iphone",
    "su7",
)
QUANTITATIVE_STATUS_QUESTION_TERMS = (
    "多少",
    "几",
    "多高",
    "多大",
    "current",
    "today",
    "latest",
    "as of",
)
EVENT_SCHEDULE_TIME_TERMS = (
    "什么时候",
    "何时",
    "哪天",
    "几号",
    "日期",
    "时间",
    "开幕",
    "开始",
    "举办",
    "开赛",
    "上映",
    "开售",
    "上市",
    "发售",
    "赛程",
    "日程",
    "when is",
    "when does",
    "when will",
    "what date",
    "schedule",
    "release date",
    "launch date",
)
EVENT_SCHEDULE_TARGET_TERMS = (
    "世界杯",
    "world cup",
    "奥运",
    "olympic",
    "总决赛",
    "决赛",
    "比赛",
    "赛事",
    "联赛",
    "nba",
    "英超",
    "欧冠",
    "super bowl",
    "演唱会",
    "音乐节",
    "展会",
    "发布会",
    "大会",
    "会议",
    "电影",
    "电视剧",
    "剧集",
    "游戏",
    "dlc",
    "高考",
    "考研",
    "考试",
    "报名",
    "春节",
    "清明",
    "端午",
    "中秋",
    "国庆",
    "假期",
    "放假",
    "spring festival",
    "thanksgiving",
)
AI_PRODUCT_TERMS = (
    "chatgpt",
    "chat gpt",
    "gpt",
    "openai",
    "claude",
    "anthropic",
    "gemini",
    "perplexity",
    "notebooklm",
    "notebook lm",
    "deepseek",
    "qwen",
    "kimi",
    "moonshot",
    "manus",
    "grok",
    "copilot",
    "cursor",
    "windsurf",
    "lovable",
    "bolt.new",
    "bolt",
    "v0",
    "google ai studio",
    "ai studio",
    "firebase studio",
    "vercel ai sdk",
    "ai sdk",
    "langchain",
    "llamaindex",
    "crewai",
    "autogen",
    "dify",
    "coze",
    "扣子",
    "cline",
    "roo code",
    "openrouter",
    "replit agent",
    "replit",
    "mistral",
    "llama",
    "sora",
    "doubao",
    "豆包",
    "hunyuan",
    "混元",
    "ernie",
    "文心",
    "通义",
    "qianwen",
    "智谱",
    "glm",
    "minimax",
    "hailuo",
    "海螺",
    "baichuan",
    "百川",
    "讯飞星火",
    "spark",
)
AI_PRODUCT_CAPABILITY_TERMS = (
    "什么是",
    "是什么",
    "是啥",
    "deep research",
    "research",
    "模型",
    "能力",
    "功能",
    "怎么用",
    "如何使用",
    "用法",
    "官方",
    "版本",
    "发布",
    "好用",
    "好用吗",
    "靠谱",
    "靠谱吗",
    "值得",
    "值得买吗",
    "对比",
    "哪个好",
    "稳定",
    "收费",
    "免费",
    "限制",
    "额度",
    "定价",
    "价格",
    "api",
    "app",
    "agent",
    "搜索",
    "联网",
    "sources",
    "citation",
    "citations",
    "what is ",
    "how to ",
    "how to use",
    "features",
    "available",
    "availability",
    "pricing",
    "cost",
    "costs",
)
STABLE_AI_CONCEPT_TARGETS = (
    "ai",
    "artificial intelligence",
    "machine learning",
    "deep learning",
    "neural network",
    "large language model",
    "llm",
)
STABLE_AI_MODIFIERS = (
    "artificial",
    "generative",
    "responsible",
    "ethical",
    "explainable",
    "narrow",
    "general",
    "strong",
    "weak",
    "agentic",
)
AI_PRODUCT_NAME_WITH_AI_RE = re.compile(r"(?i)\b([a-z][a-z0-9.+-]{2,})\s+ai\b")
AI_PRODUCT_GENERATION_RE = re.compile(r"(?i)\b[A-Za-z][A-Za-z0-9.+-]{2,}\s+(?:gen|generation)\s*[- ]?\d+(?:\.\d+)?\b")
MODEL_VERSION_RE = re.compile(
    r"(?i)\b(?:chatgpt|gpt|claude|gemini|llama|mistral|deepseek|qwen|grok|sora|o)\s*[- ]?(?:[rv]?\d+(?:\.\d+)?|sonnet\s*\d+(?:\.\d+)?)\b"
)
TECH_VERSION_RE = re.compile(r"(?i)\b[A-Za-z][A-Za-z.+#-]*\s+\d+(?:\.\d+){1,2}\b")
TECH_MAJOR_VERSION_RE = re.compile(
    r"(?i)\b(?:react|windows|ios|android|macos|python|node(?:\.js)?|next(?:\.js)?|vue|angular|chrome|edge|safari|firefox|cuda|tensorflow|pytorch|java|jdk|golang|rust)\s*(?:v|-)?\s*\d+\b"
)
ROUTE_BOUNDARY_SCHEMA_VERSION = 1
ROUTE_BOUNDARY_POLICY = "L0 search boundary"
PUBLIC_ROUTING_SCHEMA_VERSION = 1


def content_to_text(content: Any) -> str:
    """Convert OpenAI-compatible message content to text."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif isinstance(item, dict) and "text" in item:
                parts.append(str(item["text"]))
        return "\n".join(parts)
    return str(content or "")


def extract_input(payload: Dict[str, Any]) -> WorkflowInput:
    """Extract user query and chat history from supported payload shapes."""

    messages = payload.get("messages")
    if isinstance(messages, list) and messages:
        normalized = []
        for item in messages:
            if not isinstance(item, dict):
                continue
            normalized.append({"role": str(item.get("role", "")), "content": content_to_text(item.get("content", ""))})
        last_user_index = -1
        for index, item in enumerate(normalized):
            if item.get("role") == "user":
                last_user_index = index
        if last_user_index >= 0:
            user_query = normalized[last_user_index]["content"]
            return WorkflowInput(
                user_query=user_query,
                chat_history=normalized[:last_user_index] + normalized[last_user_index + 1 :],
                previous_search_context=str(payload.get("previous_search_context") or ""),
                messages=normalized,
                research_mode=_research_mode(payload),
                evidence_repair_request=_evidence_repair_request(payload),
                research_plan_preview_only=_research_plan_preview_only(payload),
                research_plan_confirmation=_research_plan_confirmation(payload),
                research_plan_confirmation_provided=_research_plan_confirmation_provided(payload),
            )
    user_query = str(payload.get("user_query") or payload.get("query") or "").strip()
    chat_history = payload.get("chat_history") if isinstance(payload.get("chat_history"), list) else []
    normalized_history = [
        {"role": str(item.get("role", "")), "content": content_to_text(item.get("content", ""))}
        for item in chat_history
        if isinstance(item, dict)
    ]
    messages = [*normalized_history, {"role": "user", "content": user_query}] if user_query else normalized_history
    return WorkflowInput(
        user_query=user_query,
        chat_history=normalized_history,
        previous_search_context=str(payload.get("previous_search_context") or ""),
        messages=messages,
        research_mode=_research_mode(payload),
        evidence_repair_request=_evidence_repair_request(payload),
        research_plan_preview_only=_research_plan_preview_only(payload),
        research_plan_confirmation=_research_plan_confirmation(payload),
        research_plan_confirmation_provided=_research_plan_confirmation_provided(payload),
    )


def _research_mode(payload: Dict[str, Any]) -> str:
    """Return a public research mode id from browser/API payloads."""

    value = payload.get("research_mode") or payload.get("researchMode") or ""
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return normalized if normalized in PUBLIC_RESEARCH_MODES else ""


def _evidence_repair_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return public evidence-repair request metadata from input payload."""

    value = payload.get("evidence_repair_request") or payload.get("evidenceRepairRequest")
    if not isinstance(value, dict):
        return {}
    allowed = {}
    for key in ("action", "label", "prompt", "started_from", "startedFrom"):
        item = value.get(key)
        if isinstance(item, (str, int, float)):
            allowed[key] = str(item)
    return allowed


def _research_plan_preview_only(payload: Dict[str, Any]) -> bool:
    """Return whether the caller only wants a report plan preview."""

    return bool(payload.get("research_plan_preview_only") or payload.get("plan_preview_only"))


def _research_plan_confirmation(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return a user-confirmed research plan from input payload."""

    value = payload.get("research_plan_confirmation") or payload.get("confirmed_research_plan")
    return value if isinstance(value, dict) else {}


def _research_plan_confirmation_provided(payload: Dict[str, Any]) -> bool:
    """Return whether the caller explicitly supplied a confirmed plan payload."""

    return isinstance(payload.get("research_plan_confirmation"), dict) or isinstance(
        payload.get("confirmed_research_plan"),
        dict,
    )


def is_probe_query(user_query: str) -> bool:
    """Return whether the query appears to probe internal implementation."""

    lower = user_query.lower()
    return any(keyword in user_query or keyword in lower for keyword in PROBE_KEYWORDS)


def is_local_direct_without_provider_query(user_query: str) -> bool:
    """Return whether the prompt can be answered locally without BYOK provider settings."""

    query = (user_query or "").strip()
    return bool(query) and (is_greeting_query(query) or is_probe_query(query))


def is_source_backed_without_llm_query(user_query: str, missing_required: List[str] | tuple[str, ...]) -> bool:
    """Return whether an ask query can still run with search configured but LLM missing."""

    query = (user_query or "").strip()
    missing = {str(item or "").strip() for item in missing_required if str(item or "").strip()}
    if not query or not missing or not missing.issubset(LLM_PROVIDER_REQUIRED_FIELDS):
        return False
    route = route_query(query, [])
    return route.level == "L0.5" and route.requires_search


def is_freshness_sensitive_query(user_query: str) -> bool:
    """Return whether the query needs fresh source-backed verification."""

    query = (user_query or "").strip()
    lower = query.lower()
    if is_quantitative_status_query(query):
        return True
    if is_event_schedule_query(query):
        return True
    if is_ai_product_like_query(query):
        return True
    if is_generic_concept_query(query):
        return False
    years = [int(match.group(1)) for match in YEAR_RE.finditer(query)]
    if any(year >= FRESHNESS_YEAR_FLOOR for year in years):
        return True
    if any(term in query or term in lower for term in FRESHNESS_TERMS):
        return True
    if any(term in query or term in lower for term in VOLATILE_ROLE_TERMS):
        if is_historical_identity_query(query):
            return False
        return True
    if any(term in lower for term in AI_PRODUCT_TERMS) and any(
        term in query or term in lower for term in AI_PRODUCT_CAPABILITY_TERMS
    ):
        return True
    return bool(MODEL_VERSION_RE.search(query) or TECH_VERSION_RE.search(query) or TECH_MAJOR_VERSION_RE.search(query))


def is_ai_product_like_query(user_query: str) -> bool:
    """Return whether a prompt names a likely modern AI product not in the alias table."""

    query = (user_query or "").strip()
    lower = query.lower()
    if not _contains_any(query, lower, AI_PRODUCT_CAPABILITY_TERMS):
        return False
    target = _generic_concept_target(query) or query
    target_lower = target.lower().strip(" \t\r\n：:，。！？!?,.、")
    if target_lower in STABLE_AI_CONCEPT_TARGETS:
        return False
    if AI_PRODUCT_GENERATION_RE.search(target):
        return True
    return _has_named_ai_product_pattern(target)


def _has_named_ai_product_pattern(text: str) -> bool:
    """Return whether text contains a product-like name followed by AI."""

    for match in AI_PRODUCT_NAME_WITH_AI_RE.finditer(text or ""):
        name = match.group(1).lower()
        if name not in STABLE_AI_MODIFIERS:
            return True
    return False


def is_quantitative_status_query(user_query: str) -> bool:
    """Return whether a numeric status query should be source-backed."""

    query = (user_query or "").strip()
    lower = query.lower()
    if not _contains_any(query, lower, VOLATILE_METRIC_TARGET_TERMS):
        return False
    if lower.startswith(("什么是", "what is ", "explain ")):
        return False
    return _contains_any(query, lower, QUANTITATIVE_STATUS_QUESTION_TERMS)


def is_event_schedule_query(user_query: str) -> bool:
    """Return whether an event, release, or holiday schedule needs sources."""

    query = (user_query or "").strip()
    lower = query.lower()
    return _contains_any(query, lower, EVENT_SCHEDULE_TIME_TERMS) and _contains_any(
        query,
        lower,
        EVENT_SCHEDULE_TARGET_TERMS,
    )


def is_generic_concept_query(user_query: str) -> bool:
    """Return whether a query asks for a stable concept definition."""

    query = (user_query or "").strip()
    lower = query.lower()
    target = _generic_concept_target(query)
    if not target:
        return False

    target_lower = target.lower()
    years = [int(match.group(1)) for match in YEAR_RE.finditer(query)]
    if any(year >= FRESHNESS_YEAR_FLOOR for year in years):
        return False
    if any(term in query or term in lower for term in CONCEPT_FRESHNESS_BLOCK_TERMS):
        return False
    if MODEL_VERSION_RE.search(query) or TECH_VERSION_RE.search(query) or TECH_MAJOR_VERSION_RE.search(query):
        return False
    if is_ai_product_like_query(query):
        return False
    if any(term in lower for term in AI_PRODUCT_TERMS):
        return False

    if target in GENERIC_REGULATORY_CONCEPTS or target_lower in GENERIC_REGULATORY_CONCEPTS:
        return True
    if _contains_any(target, target_lower, REGULATORY_TARGET_TERMS) and _contains_any(
        target,
        target_lower,
        REGULATORY_ENTITY_TERMS,
    ):
        return False
    if _contains_any(target, target_lower, VOLATILE_METRIC_TARGET_TERMS) and _contains_any(
        target,
        target_lower,
        METRIC_ENTITY_TERMS,
    ):
        return False
    return True


def _generic_concept_target(user_query: str) -> str:
    """Extract the definition target from a simple concept query."""

    query = (user_query or "").strip().strip(" \t\r\n，。！？!?,.、")
    lower = query.lower()
    for prefix in GENERIC_CONCEPT_PREFIXES:
        if lower.startswith(prefix):
            return query[len(prefix) :].strip(" \t\r\n：:，。！？!?,.、")
    for suffix in GENERIC_CONCEPT_SUFFIXES:
        if lower.endswith(suffix):
            return query[: -len(suffix)].strip(" \t\r\n：:，。！？!?,.、")
    return ""


def _contains_any(text: str, lower: str, terms: tuple[str, ...]) -> bool:
    """Return whether text contains any case-insensitive mixed-language term."""

    return any(term in text or term in lower for term in terms)


def is_historical_identity_query(user_query: str) -> bool:
    """Return whether role-like wording refers to a stable historical identity."""

    query = (user_query or "").strip()
    lower = query.lower()
    return any(term in query or term in lower for term in HISTORICAL_IDENTITY_TERMS)


def is_stable_direct_query(user_query: str) -> bool:
    """Return whether the query is safe for an offline direct answer."""

    query = (user_query or "").strip()
    lower = query.lower()
    if not query or is_freshness_sensitive_query(query):
        return False
    if is_generic_concept_query(query):
        return True

    stable_prefixes = (
        "什么是",
        "介绍一下",
        "解释",
        "解释一下",
        "怎么理解",
        "为什么",
        "为何",
        "如何理解",
        "怎样理解",
        "原理",
        "how to ",
        "what is ",
        "how does ",
        "why is ",
        "why does ",
        "explain ",
    )
    stable_suffixes = (
        "怎么用",
        "是什么",
        "是啥",
        "是多少",
        "是谁",
        "存在吗",
        "有什么区别",
        "的原理",
    )
    stable_infixes = (
        "为什么",
        "为何",
        "什么时候",
        "何时",
        "哪一年",
        "谁发明",
    )
    stable_english_prefixes = (
        "who is ",
        "who was ",
        "when was ",
        "when did ",
        "who invented ",
    )
    return (
        is_greeting_query(query)
        or lower.startswith(stable_prefixes)
        or lower.startswith(stable_english_prefixes)
        or query.startswith(stable_prefixes)
        or query.endswith(stable_suffixes)
        or any(term in query for term in stable_infixes)
    )


def is_greeting_query(user_query: str) -> bool:
    """Return whether the input is only lightweight chat with no research intent."""

    query = (user_query or "").strip().lower()
    query = re.sub(r"[\s，。！？!?.~～、,.]+", "", query)
    if not query:
        return False
    greetings = {
        "你好",
        "您好",
        "嗨",
        "哈喽",
        "hello",
        "hi",
        "hey",
        "早上好",
        "上午好",
        "中午好",
        "下午好",
        "晚上好",
        "谢谢",
        "谢谢你",
        "感谢",
        "感谢你",
        "多谢",
        "辛苦了",
        "好的",
        "好的谢谢",
        "收到",
        "收到谢谢",
        "明白了",
        "明白了谢谢",
        "懂了",
        "懂了谢谢",
        "ok",
        "okay",
        "thanks",
        "thankyou",
    }
    greeting_suffixes = ("呀", "啊", "哦", "哈")
    if query in greetings:
        return True
    return any(query == f"{greeting}{suffix}" for greeting in greetings for suffix in greeting_suffixes)


def route_boundary(decision: RouteDecision) -> Dict[str, Any]:
    """Return a user-facing explanation of the L0 search boundary."""

    return normalize_route_boundary(
        {
            "schema_version": ROUTE_BOUNDARY_SCHEMA_VERSION,
            "level": decision.level,
            "label": decision.label,
            "requires_search": decision.requires_search,
            "policy": ROUTE_BOUNDARY_POLICY,
            "boundary_label": decision.boundary_label,
            "reason": decision.boundary_reason,
            "user_notice": decision.user_notice,
        },
        routing=decision,
    )


def normalize_route_boundary(value: Any, *, routing: Any = None) -> Dict[str, Any]:
    """Return a safe public route-boundary envelope."""

    raw = value if isinstance(value, dict) else {}
    route = _routing_dict(routing)
    level = _safe_text(raw.get("level") or route.get("level") or "")
    label = _safe_text(raw.get("label") or route.get("label") or _default_route_label(level))
    requires_search = bool(raw.get("requires_search") if "requires_search" in raw else _default_requires_search(route, level))
    boundary_label = _safe_text(raw.get("boundary_label") or raw.get("boundaryLabel") or _default_boundary_label(level, requires_search))
    reason = _safe_text(raw.get("reason") or _default_boundary_reason(level, requires_search))
    user_notice = _safe_text(raw.get("user_notice") or raw.get("userNotice") or _default_user_notice(level, requires_search))
    return {
        "schema_version": ROUTE_BOUNDARY_SCHEMA_VERSION,
        "level": level,
        "label": label,
        "requires_search": requires_search,
        "policy": ROUTE_BOUNDARY_POLICY,
        "boundary_label": boundary_label,
        "reason": reason,
        "user_notice": user_notice,
    }


def normalize_public_routing(value: Any, *, route_boundary: Any = None) -> Dict[str, Any]:
    """Return a public routing summary without internal decision fields."""

    route = _routing_dict(value)
    boundary = route_boundary if isinstance(route_boundary, dict) else {}
    level = _safe_text(route.get("level") or boundary.get("level") or "")
    label = _safe_text(route.get("label") or boundary.get("label") or _default_route_label(level))
    requires_search = bool(
        route.get("requires_search") if "requires_search" in route else boundary.get("requires_search", _default_requires_search(route, level))
    )
    return {
        "schema_version": PUBLIC_ROUTING_SCHEMA_VERSION,
        "level": level,
        "label": label,
        "requires_search": requires_search,
        "search_branches": _safe_non_negative_int(route.get("search_branches")),
        "section_count": _safe_non_negative_int(route.get("section_count")),
        "effort_level": _safe_non_negative_int(route.get("effort_level")),
    }


def route_query(user_query: str, chat_history: List[Dict[str, str]]) -> RouteDecision:
    """Route a query into L0, L0.5, L1, L2, or L3."""

    query = (user_query or "").strip()
    lower = query.lower()
    if not query or is_probe_query(query):
        return _decision("L0", "DIRECT_REPLY", 0, "直接回答", 0, 0, False, False, "安全直答", "空输入或安全防护问题不会启动搜索。")

    strong_report = ("报告", "研究报告", "深度研究", "简报", "出具", "撰写", "write a report", "research report")
    followup_starts = ("那", "那么", "还有", "继续", "它", "这个", "这", "why", "what about")
    freshness_sensitive = is_freshness_sensitive_query(query)

    if is_greeting_query(query):
        return _decision("L0", "DIRECT_REPLY", 0, "直接回答", 0, 0, False, False, "稳定直答", "这是问好，不依赖当前外部状态。")

    if chat_history and (query.startswith(followup_starts) or lower.startswith(followup_starts)) and not any(
        signal in query or signal in lower for signal in strong_report
    ):
        return _decision("L0.5", "QUICK_SEARCH", 1, "搜索问答", 1, 0, False, False, "追问查证", "多轮追问默认保留查证边界，避免回落到 L0 模型记忆。")

    if is_stable_direct_query(query):
        return _decision("L0", "DIRECT_REPLY", 0, "直接回答", 0, 0, False, False, "稳定直答", "这是稳定常识或通用方法问题，不依赖当前外部状态。")

    if any(signal in query or signal in lower for signal in strong_report):
        if "深度研究" in query or "deep research" in lower:
            return _decision("L3", "MAIN_WORKFLOW", 3, "深度报告", 4, 4, True, True, "报告流查证", "报告意图需要多来源搜索、验证和事实核查。")
        if "对比" in query or "竞争" in query or "research report" in lower:
            return _decision("L2", "MAIN_WORKFLOW", 2, "标准报告", 2, 3, True, False, "报告流查证", "报告意图需要多来源搜索和结构化整理。")
        return _decision("L1", "MAIN_WORKFLOW", 1, "简报", 1, 2, False, False, "报告流查证", "报告意图需要至少一轮来源检索。")

    reason = "问题涉及当前事实、版本、身份、价格、新闻或产品状态。" if freshness_sensitive else "非稳定解释类任务默认先查证来源。"
    return _decision("L0.5", "QUICK_SEARCH", 1, "搜索问答", 1, 0, False, False, "需要查证", reason)


def _decision(
    level: str,
    routing_decision: str,
    effort_level: int,
    label: str,
    search_branches: int,
    section_count: int,
    enable_validation: bool,
    enable_fact_check: bool,
    boundary_label: str,
    boundary_reason: str,
) -> RouteDecision:
    """Build a route decision with public route-boundary metadata."""

    requires_search = level != "L0" or search_branches > 0
    return RouteDecision(
        level,
        routing_decision,
        effort_level,
        label,
        search_branches,
        section_count,
        enable_validation,
        enable_fact_check,
        requires_search,
        boundary_label,
        boundary_reason,
        _default_user_notice(level, requires_search),
    )


def _routing_dict(value: Any) -> Dict[str, Any]:
    """Return a dict-shaped route decision."""

    if isinstance(value, RouteDecision):
        return {
            "level": value.level,
            "label": value.label,
            "search_branches": value.search_branches,
            "requires_search": value.requires_search,
        }
    return value if isinstance(value, dict) else {}


def _default_requires_search(route: Dict[str, Any], level: str) -> bool:
    """Infer whether the route requires source-backed work."""

    if "requires_search" in route:
        return bool(route.get("requires_search"))
    if level and level != "L0":
        return True
    try:
        return int(route.get("search_branches") or 0) > 0
    except (TypeError, ValueError):
        return False


def _default_route_label(level: str) -> str:
    """Return a compact route label."""

    return {"L0": "直接回答", "L0.5": "搜索问答", "L1": "简报", "L2": "标准报告", "L3": "深度报告"}.get(level, "研究流程")


def _default_boundary_label(level: str, requires_search: bool) -> str:
    """Return a public boundary label."""

    if not requires_search:
        return "稳定直答"
    if level in {"L1", "L2", "L3"}:
        return "报告流查证"
    return "需要查证"


def _default_boundary_reason(level: str, requires_search: bool) -> str:
    """Return a public route-boundary reason."""

    if not requires_search:
        return "这是稳定常识或通用方法问题，不依赖当前外部状态。"
    if level in {"L1", "L2", "L3"}:
        return "报告意图需要来源检索、分析和质量门禁。"
    return "问题可能依赖当前事实、版本、身份、价格、新闻或产品状态。"


def _default_user_notice(level: str, requires_search: bool) -> str:
    """Return a low-burden user-facing notice for the route boundary."""

    if not requires_search:
        return "稳定常识会直接回答；涉及最新状态、官方版本、价格、身份、新闻或政策的问题会先查证来源。"
    if level in {"L1", "L2", "L3"}:
        return "这类任务不会走纯本地直答；会进入本地研究流程，先查证来源再组织报告。"
    return "这类问题不会走纯本地直答；会先查证来源，再基于来源回答。"


def _safe_text(value: Any, *, limit: int = 180) -> str:
    """Return compact public text for route-boundary fields."""

    text = str(value or "").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def _safe_non_negative_int(value: Any) -> int:
    """Return a public non-negative integer for route summary counters."""

    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0
