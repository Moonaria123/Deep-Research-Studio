"""Run-scoped external search candidates for demand-driven retrieval.

The prefetch decision in this module is deliberately weaker than grounding.  It
uses the T48 eligibility rules to decide whether a search snippet is worth using
or fetching, but final evidence must still pass the normal orchestrator gates.
"""

from __future__ import annotations

import hashlib
import re
import urllib.parse
from dataclasses import dataclass, replace
from typing import Literal, Sequence

from .citations import semantic_binding_conflict
from .evidence_eligibility import ResearchIntent, assess_evidence
from .retrieval import EvidenceItem

ContentBasis = Literal["snippet", "fetched"]
PrefetchDecision = Literal["candidate", "rejected", "unknown"]

_TRACKING_QUERY_KEYS = frozenset({"fbclid", "gclid", "mc_cid", "mc_eid"})
_TRUNCATED_SNIPPET = re.compile(r"(?:\.{3}|\u2026|\[\s*\.\.\.\s*\])\s*$")
_NONCONTIGUOUS_EXCERPT_GAP = re.compile(r"(?:\.{3}|\u2026|\[\s*\.\.\.\s*\])")
_SNIPPET_TOKEN = re.compile(r"[\w\u3400-\u9fff]+", re.UNICODE)
_FOCUS_TOKEN = re.compile(r"[a-z0-9]+|[\u3400-\u9fff]", re.IGNORECASE)
_SOURCE_ATTRIBUTION_PREFIX = re.compile(
    r"^\s*(?:(?:according\s+to|per)\s+(?P<latin>[^,，:：]{1,80})|"
    r"(?:根据|据)\s*(?P<cjk>[^,，:：]{1,40}))\s*[,，:：]\s*",
    re.IGNORECASE,
)
_SOURCE_ATTRIBUTION_GENERIC_TOKENS = frozenset(
    {
        "a",
        "an",
        "the",
        "official",
        "documentation",
        "documents",
        "docs",
        "guidance",
        "website",
        "site",
        "source",
        "sources",
    }
)
_FOCUS_EXCERPT_STOPWORDS = frozenset(
    {
        "according",
        "answer",
        "cited",
        "concise",
        "does",
        "get",
        "guidance",
        "how",
        "many",
        "most",
        "much",
        "need",
        "official",
        "per",
        "should",
        "source",
        "what",
        "which",
        "with",
    }
)
_PROBLEM_SOLUTION_QUERY = re.compile(
    r"\bwhat\s+problem\s+(?:does|do|did)\b.{0,120}\bsolve\b"
    r"|\bproblem\b.{0,120}\bsolves?\b"
    r"|(?:解决|应对).{0,40}(?:什么|哪些)?(?:问题|痛点)",
    re.IGNORECASE,
)
_PROBLEM_SOLUTION_OVERVIEW = re.compile(
    r"\b(?:this|the)\s+(?:module|library|package|tool|system|service|framework|api|product)\b"
    r"(?!['’])"
    r".{0,120}\b(?:solves?|addresses|offers?|provides?|enables?|allows?|helps?)\b"
    r"|(?:该|这个|此)(?:模块|库|软件包|工具|系统|服务|框架|接口|产品).{0,80}"
    r"(?:解决|应对|提供|支持|允许|帮助)",
    re.IGNORECASE,
)
_NEGATED_PROBLEM_SOLUTION_ACTION = re.compile(
    r"\b(?:do(?:es)?\s+not|don['’]t|doesn['’]t|cannot|can['’]t|never)\s+"
    r"(?:\w+\s+){0,2}(?:solves?|addresses|offers?|provides?|enables?|allows?|helps?)\b",
    re.IGNORECASE,
)
_CAUSE_QUERY = re.compile(
    r"\bwhat\s+(?:primarily\s+)?causes?\b|\bwhat\s+is\s+the\s+cause\b|什么(?:主要)?导致|成因",
    re.IGNORECASE,
)
_PRIMARY_CAUSE_QUERY = re.compile(
    r"\bwhat\s+primarily\s+causes?\b|"
    r"\bwhat\s+is\s+the\s+(?:primary|main|dominant|principal)\s+cause\b|"
    r"(?:主要|首要|主导)(?:原因|成因)",
    re.IGNORECASE,
)
_CAUSE_ANSWER = re.compile(
    r"\b(?:causes|caused\s+by|creates|created\s+by|results?\s+from|is\s+due\s+to|because)\b"
    r"|(?:由|因为|源于|导致|造成)",
    re.IGNORECASE,
)
_PRIMARY_CAUSE_ANSWER = re.compile(
    r"\b(?:primary|main|dominant|principal|leading)\s+(?:cause|driver|factor|force)\b|"
    r"\bprimarily\s+(?:caused|driven)\s+by\b|"
    r"(?:主要|首要|主导)(?:原因|成因|因素|力量)",
    re.IGNORECASE,
)
_PURPOSE_QUERY = re.compile(
    r"\bwhat\s+is\s+(?:the\s+)?purpose\b|\bwhat\s+purpose\b|\bpurpose\s+of\b|(?:目的|用途)",
    re.IGNORECASE,
)
_PURPOSE_ANSWER = re.compile(
    r"\b(?:help(?:s|ed|ing)?|intend(?:s|ed|ing)?|design(?:s|ed)?|aim(?:s|ed)?|"
    r"used\s+(?:to|for)|enable(?:s|d)?|allow(?:s|ed)?)\b|"
    r"\bpurpose\s+is\s+to\b|"
    r"(?:旨在|用于|帮助|支持|使得|目的)",
    re.IGNORECASE,
)
_DEFINITION_QUERY = re.compile(
    r"\b(?:definition|define)\b|"
    r"\bwhat\s+(?:is|are)\s+(?!the\s+(?:purpose|cause|reason)\b)|"
    r"\bwhat\s+does\b.{1,100}\b(?:mean|measure)\b|"
    r"(?:定义|含义|(?:衡量|测量)什么|是什么)",
    re.IGNORECASE,
)
_DEFINITION_SUBJECT = re.compile(
    r"\bwhat\s+(?:is|are)\s+(?:the\s+definition\s+of\s+)?(?P<subject>[^?]{2,100})\??$|"
    r"\bwhat\s+does\s+(?P<does_subject>[^?]{2,100}?)\s+(?:mean|measure)\b",
    re.IGNORECASE,
)
_DEFINITION_ANSWER = re.compile(
    r"\b(?:defines?|defined)\b.{0,100}\bas\b|"
    r"\b(?:is|are|refers?\s+to|means?|measures?|indicates?|represents?|instantiates?|"
    r"supports?|provides?|offers?|enables?)\b|"
    r"(?:定义为|是指|指的是|意味着|衡量|测量)",
    re.IGNORECASE,
)
_OVERVIEW_QUERY = re.compile(
    r"\b(?:overview|introduction|use cases?)\b|(?:概述|简介|用途|使用场景)",
    re.IGNORECASE,
)
_OVERVIEW_ANSWER = re.compile(
    r"\b(?:is|are|supports?|provides?|offers?|enables?|allows?|features?|implements?|uses?)\b|"
    r"(?:是|支持|提供|允许|具有|实现|用于)",
    re.IGNORECASE,
)
_OVERVIEW_FOCUS_STOPWORDS = frozenset(
    {
        "and",
        "cases",
        "introduction",
        "its",
        "of",
        "overview",
        "the",
        "use",
        "uses",
    }
)
_QUERY_SYNTAX_QUERY = re.compile(
    r"\b(?:query\s+syntax|search\s+patterns?|match\s+(?:query|expression|syntax))\b|"
    r"(?:查询语法|搜索模式|匹配语法)",
    re.IGNORECASE,
)
_QUERY_BOOLEAN_OPERATOR = re.compile(r"\b(?:AND|OR|NOT)\b")
_RANKING_QUERY = re.compile(
    r"\b(?:ranking|relevance\s+scor(?:e|es|ing))\b|(?:排序|相关性评分)",
    re.IGNORECASE,
)
_RANKING_EXPLANATION = re.compile(
    r"\b(?:returns?|contains?|indicat(?:e|es|ing)|reflect(?:s|ing)?|"
    r"better\s+matches?|smaller\s+values?|worse\s+matches?|"
    r"sort(?:s|ed|ing)?\s+(?:results?|matches?)|score(?:s|d|ing)?)\b|"
    r"(?:返回|表示|反映|匹配越好|数值越小|排序结果|相关性得分)",
    re.IGNORECASE,
)
_NONASSERTIVE_QUESTION = re.compile(r"^\s*(?:#{1,6}\s+)?[^\r\n]{1,240}[?？]\s*#*\s*$")
_NONASSERTIVE_HEADING = re.compile(r"^\s*#{1,6}\s+\S.*?\s*#*\s*$")
_NORMATIVE_QUERY = re.compile(
    r"\b(?:should|guidance|recommendations?)\b|\bhow\s+(?:much|many)\b|"
    r"(?:应该|应当|建议|多少|多久)",
    re.IGNORECASE,
)
_NORMATIVE_DIRECT_TITLE = re.compile(
    r"\b(?:guidance|guidelines?|recommendations?|how\s+(?:much|many)|needs?)\b|"
    r"(?:指南|指导|建议|需要多少)",
    re.IGNORECASE,
)
_NORMATIVE_OVERVIEW_TITLE = re.compile(
    r"\babout\b|(?:概览|简介)",
    re.IGNORECASE,
)
_NORMATIVE_SUPPORT_TITLE = re.compile(
    r"\b(?:tips?|enough)\b|(?:须知|贴士)",
    re.IGNORECASE,
)
_NORMATIVE_STATISTICS_TITLE = re.compile(
    r"\b(?:faststats|statistics|stats|survey|study|data[\s_-]*brief|research)\b|"
    r"(?:统计|调查|研究|数据简报)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExternalCandidate:
    """Immutable search/fetch candidate; never a claim-level coverage result."""

    candidate_id: str
    plan_id: str
    query: str
    title: str
    url: str
    canonical_url: str
    provenance_root: str
    snippet: str
    source: str = ""
    provider: str = ""
    fetch_provider: str = ""
    tier: str = ""
    rank: int = 0
    content_basis: ContentBasis = "snippet"
    prefetch_decision: PrefetchDecision = "unknown"
    relevance: str = "uncertain"
    authority_status: str = "not_requested"
    reason_codes: tuple[str, ...] = ()
    matched_anchors: tuple[str, ...] = ()
    published_at: float | None = None
    updated_at: float | None = None
    effective_at: float | None = None
    retrieved_at: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible audit view."""

        return {
            "candidate_id": self.candidate_id,
            "plan_id": self.plan_id,
            "query": self.query,
            "title": self.title,
            "url": self.url,
            "canonical_url": self.canonical_url,
            "provenance_root": self.provenance_root,
            "snippet": self.snippet,
            "source": self.source,
            "provider": self.provider,
            "fetch_provider": self.fetch_provider,
            "tier": self.tier,
            "rank": self.rank,
            "content_basis": self.content_basis,
            "prefetch_decision": self.prefetch_decision,
            "relevance": self.relevance,
            "authority_status": self.authority_status,
            "reason_codes": list(self.reason_codes),
            "matched_anchors": list(self.matched_anchors),
            "published_at": self.published_at,
            "updated_at": self.updated_at,
            "effective_at": self.effective_at,
            "retrieved_at": self.retrieved_at,
        }


def title_query_phrase_span(candidate: ExternalCandidate) -> int:
    """Return the longest contiguous query phrase represented in the title.

    Search providers often omit snippets for authoritative pages.  When several
    same-identity candidates therefore remain equally unknown, the title phrase
    match is a deterministic signal for which single Fast-mode fetch is most
    likely to answer the user's actual question.
    """

    query_tokens = [token.casefold() for token in _FOCUS_TOKEN.findall(candidate.query)]
    title_tokens = [token.casefold() for token in _FOCUS_TOKEN.findall(candidate.title)]
    if not query_tokens or not title_tokens:
        return 0

    previous = [0] * (len(title_tokens) + 1)
    best = 0
    for query_token in query_tokens:
        current = [0] * (len(title_tokens) + 1)
        for index, title_token in enumerate(title_tokens, start=1):
            if query_token == title_token:
                current[index] = previous[index - 1] + 1
                best = max(best, current[index])
        previous = current
    return best


def title_intent_fit_score(candidate: ExternalCandidate) -> int:
    """Rank answer-oriented pages ahead of statistics for normative questions."""

    if not _NORMATIVE_QUERY.search(candidate.query):
        return 0
    identity = f"{candidate.title} {candidate.canonical_url or candidate.url}"
    score = 0
    if _NORMATIVE_DIRECT_TITLE.search(identity):
        score += 3
    if _NORMATIVE_OVERVIEW_TITLE.search(identity):
        score += 3
    if _NORMATIVE_SUPPORT_TITLE.search(identity):
        score += 1
    if _NORMATIVE_STATISTICS_TITLE.search(identity):
        score -= 3
    return score


def canonicalize_external_url(value: object) -> str:
    """Return a conservative HTTP(S) document identity or an empty string."""

    raw = str(value or "").strip()
    try:
        parsed = urllib.parse.urlsplit(raw)
        scheme = parsed.scheme.lower()
        hostname = (parsed.hostname or "").encode("idna").decode("ascii").lower()
        port = parsed.port
    except (UnicodeError, ValueError):
        return ""
    if scheme not in {"http", "https"} or not hostname:
        return ""
    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    netloc = hostname if port is None or default_port else f"{hostname}:{port}"
    path = parsed.path or "/"
    query = urllib.parse.urlencode(
        sorted(
            (
                (key, item)
                for key, item in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
                if key.lower() not in _TRACKING_QUERY_KEYS and not key.lower().startswith("utm_")
            ),
            key=lambda pair: (pair[0], pair[1]),
        ),
        doseq=True,
    )
    return urllib.parse.urlunsplit((scheme, netloc, path, query, ""))


def snippet_is_truncated(value: object) -> bool:
    """Return whether a provider snippet contains a visible omission marker."""

    text = str(value or "").strip()
    return not text or bool(_NONCONTIGUOUS_EXCERPT_GAP.search(text))


def _source_attribution_tokens(intent: ResearchIntent) -> tuple[str, ...]:
    """Return a named source identity from an explicit attribution prefix."""

    match = _SOURCE_ATTRIBUTION_PREFIX.match(intent.research_focus)
    if match is None:
        return ()
    raw = str(match.group("latin") or match.group("cjk") or "").strip()
    if not raw:
        return ()
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9-]*|[\u3400-\u9fff]{2,}", raw)
    meaningful = [
        token
        for token in raw_tokens
        if token.lower() not in _SOURCE_ATTRIBUTION_GENERIC_TOKENS
    ]
    if not meaningful:
        return ()
    named_identity = any(
        re.fullmatch(r"[\u3400-\u9fff]{2,}", token)
        or (len(token) >= 2 and token.isupper())
        or (token[:1].isupper() and token[1:].islower())
        for token in meaningful
    )
    if not named_identity:
        return ()
    return tuple(token.lower() for token in meaningful)


def _source_attribution_identity_match(
    candidate: ExternalCandidate,
    intent: ResearchIntent,
) -> bool | None:
    """Match a requested named source against canonical provenance identity."""

    target_tokens = _source_attribution_tokens(intent)
    if not target_tokens:
        return None
    hostname = ""
    try:
        hostname = urllib.parse.urlsplit(
            candidate.canonical_url or candidate.url
        ).hostname or ""
    except ValueError:
        hostname = ""
    identity = " ".join(part for part in (hostname, candidate.source) if part).lower()
    if not identity:
        return None
    identity_tokens = set(re.findall(r"[a-z][a-z0-9-]*|[\u3400-\u9fff]{2,}", identity))
    compact_identity = re.sub(r"[^a-z0-9\u3400-\u9fff]+", "", identity)
    compact_target = "".join(target_tokens)
    if compact_target and compact_target in compact_identity:
        return True
    if all(token in identity_tokens for token in target_tokens):
        return True
    if len(target_tokens) >= 2:
        acronym = "".join(token[0] for token in target_tokens if token)
        if len(acronym) >= 2 and acronym in identity_tokens:
            return True
    return False


def build_external_candidate(
    *,
    plan_id: str,
    query: str,
    title: str,
    url: str,
    snippet: str,
    intent: ResearchIntent,
    canonical_url: str | None = None,
    source: str = "",
    provider: str = "",
    tier: str = "",
    rank: int = 0,
    content_basis: ContentBasis = "snippet",
    published_at: float | None = None,
    updated_at: float | None = None,
    effective_at: float | None = None,
    retrieved_at: float | None = None,
) -> ExternalCandidate:
    """Build and conservatively prefilter one external candidate."""

    normalized_url = canonicalize_external_url(url)
    normalized_canonical = canonicalize_external_url(canonical_url) if canonical_url else normalized_url
    provenance_root = normalized_canonical or normalized_url
    digest = hashlib.sha256(
        f"{plan_id}\0{query}\0{provenance_root or url}\0{max(0, int(rank))}".encode("utf-8")
    ).hexdigest()[:20]
    candidate = ExternalCandidate(
        candidate_id=f"ext_{digest}",
        plan_id=str(plan_id),
        query=str(query),
        title=str(title or normalized_url or url),
        url=normalized_url or str(url or "").strip(),
        canonical_url=normalized_canonical,
        provenance_root=provenance_root,
        snippet=str(snippet or "").strip(),
        source=str(source or ""),
        provider=str(provider or ""),
        tier=str(tier or ""),
        rank=max(0, int(rank)),
        content_basis=content_basis,
        published_at=published_at,
        updated_at=updated_at,
        effective_at=effective_at,
        retrieved_at=retrieved_at,
    )
    return prefilter_external_candidate(
        candidate,
        intent,
        identity_mismatch=bool(
            normalized_url
            and normalized_canonical
            and not _same_document_host(normalized_url, normalized_canonical)
        ),
    )


def candidate_from_search_hit(
    hit: object,
    *,
    plan_id: str,
    query: str,
    intent: ResearchIntent,
) -> ExternalCandidate:
    """Normalize an adapter SearchHit without importing adapter types here."""

    metadata = getattr(hit, "metadata", {})
    metadata = dict(metadata) if isinstance(metadata, dict) else {}
    canonical_hint = next(
        (
            metadata.get(key)
            for key in ("canonical_url", "canonicalUrl", "source_url", "sourceURL")
            if metadata.get(key)
        ),
        None,
    )
    return build_external_candidate(
        plan_id=plan_id,
        query=query,
        title=str(getattr(hit, "title", "") or ""),
        url=str(getattr(hit, "url", "") or ""),
        snippet=str(getattr(hit, "snippet", "") or ""),
        intent=intent,
        canonical_url=str(canonical_hint) if canonical_hint else None,
        source=str(getattr(hit, "source", "") or ""),
        provider=str(metadata.get("search_provider") or ""),
        tier=str(metadata.get("fallback_tier") or ""),
        rank=int(getattr(hit, "rank", 0) or 0),
        published_at=_timestamp(metadata.get("published_at")),
        updated_at=_timestamp(metadata.get("updated_at")),
        effective_at=_timestamp(metadata.get("effective_at")),
        retrieved_at=_timestamp(metadata.get("retrieved_at")),
    )


def prefilter_external_candidate(
    candidate: ExternalCandidate,
    intent: ResearchIntent,
    *,
    identity_mismatch: bool = False,
) -> ExternalCandidate:
    """Apply T48 relevance/authority rules as a non-grounding prefetch filter."""

    probe = EvidenceItem(
        chunk_id=-1,
        document_id=candidate.provenance_root or candidate.url,
        url=candidate.canonical_url or candidate.url,
        title=candidate.title,
        text=candidate.snippet,
        score=1.0,
        provenance_refs=[candidate.provenance_root] if candidate.provenance_root else [],
        published_at=candidate.published_at,
        updated_at=candidate.updated_at,
        effective_at=candidate.effective_at,
        retrieved_at=candidate.retrieved_at,
    )
    outcome = assess_evidence(intent, [probe]).outcomes[0]
    decision: PrefetchDecision = (
        "candidate" if outcome.decision == "eligible" else outcome.decision
    )
    reasons = list(outcome.reason_codes)
    # A missing/short search snippet says nothing reliable about the page itself.
    # T48 correctly rejects it as final citable evidence, while this weaker prefetch
    # gate must keep it fetchable instead of confusing absence with off-topic proof.
    if "not_citable" in reasons:
        decision = "unknown"
    if not candidate.canonical_url:
        reasons.append("canonical_identity_unavailable")
        if decision == "candidate":
            decision = "unknown"
    if identity_mismatch:
        reasons.append("canonical_domain_mismatch")
        if decision == "candidate":
            decision = "unknown"
    if candidate.content_basis == "snippet" and snippet_is_truncated(candidate.snippet):
        reasons.append("snippet_truncated")
        if decision == "candidate":
            decision = "unknown"
    if (
        candidate.content_basis == "snippet"
        and len(_SNIPPET_TOKEN.findall(candidate.snippet)) < 3
    ):
        reasons.append("snippet_insufficient")
        if decision != "rejected" or "not_citable" in reasons:
            decision = "unknown"
    attribution_tokens = _source_attribution_tokens(intent)
    if attribution_tokens:
        attribution_match = _source_attribution_identity_match(candidate, intent)
        if attribution_match is False:
            decision = "rejected"
            reasons.append("source_attribution_identity_mismatch")
        elif attribution_match is None:
            decision = "unknown"
            reasons.append("source_attribution_identity_unverified")
        else:
            reasons.append("source_attribution_identity_matched")
            if candidate.content_basis == "snippet":
                # A search-result description is not enough to bind a claim to a
                # specifically named publisher. Fetch and assess the document body.
                decision = "unknown"
                reasons.append("source_attribution_requires_fetched_body")
    return replace(
        candidate,
        prefetch_decision=decision,
        relevance=outcome.relevance,
        authority_status=outcome.authority_status,
        reason_codes=tuple(dict.fromkeys(reasons)),
        matched_anchors=tuple(outcome.matched_anchors),
    )


def bind_fetched_content(
    candidate: ExternalCandidate,
    *,
    intent: ResearchIntent,
    document_url: str,
    title: str,
    text: str,
    canonical_url: str | None = None,
    fetch_provider: str = "",
) -> ExternalCandidate:
    """Return a fetched candidate while retaining its stable run-scoped id."""

    normalized_document = canonicalize_external_url(document_url)
    normalized_canonical = canonicalize_external_url(canonical_url) if canonical_url else normalized_document
    mismatch = bool(
        candidate.canonical_url
        and normalized_canonical
        and not _same_document_host(candidate.canonical_url, normalized_canonical)
    )
    fetched = replace(
        candidate,
        title=str(title or candidate.title),
        url=normalized_document or candidate.url,
        canonical_url=normalized_canonical or candidate.canonical_url,
        provenance_root=(
            normalized_canonical
            if normalized_canonical and not mismatch
            else candidate.provenance_root
        ),
        snippet=str(text or "").strip(),
        content_basis="fetched",
        fetch_provider=str(fetch_provider or ""),
        prefetch_decision="unknown",
        reason_codes=(),
        matched_anchors=(),
    )
    return prefilter_external_candidate(fetched, intent, identity_mismatch=mismatch)


def fetched_body_may_contain_focused_evidence(candidate: ExternalCandidate) -> bool:
    """Allow unknown full bodies to reach query-focused assessment, never final grounding."""

    return bool(candidate.snippet.strip()) and candidate.prefetch_decision != "rejected"


def select_fetched_evidence_excerpt(
    intent: ResearchIntent,
    pieces: Sequence[str],
    *,
    document_url: str = "",
    document_title: str = "",
    max_chars: int = 1200,
    citation_chars: int = 240,
) -> tuple[str, int]:
    """Select a focus-relevant sentence window instead of page-leading boilerplate.

    The fetched body remains available for cache/index persistence.  Only this
    query-scoped excerpt becomes the claim-level EvidenceItem, so Citation.snippet
    starts with the passage that actually satisfied the relevance gate.
    """

    content_focus = _SOURCE_ATTRIBUTION_PREFIX.sub("", intent.research_focus).strip()
    excerpt_intent = (
        replace(intent, research_focus=content_focus)
        if content_focus and content_focus != intent.research_focus
        else intent
    )
    content_intent = replace(excerpt_intent, source_requirement="none")
    decision_rank = {"rejected": 0, "unknown": 1, "eligible": 2}
    relevance_rank = {"unrelated": 0, "unknown": 1, "weak": 2, "strong": 3}
    definition_question = bool(_DEFINITION_QUERY.search(excerpt_intent.research_focus))
    purpose_question = bool(_PURPOSE_QUERY.search(excerpt_intent.research_focus))
    overview_question = bool(_OVERVIEW_QUERY.search(excerpt_intent.research_focus))
    query_syntax_question = bool(_QUERY_SYNTAX_QUERY.search(excerpt_intent.research_focus))
    ranking_question = bool(_RANKING_QUERY.search(excerpt_intent.research_focus))
    default_sentence_count = 2 if any(
        (overview_question, query_syntax_question, ranking_question)
    ) else 1
    desired_sentence_count = max(
        1,
        min(3, int(excerpt_intent.sentence_count or default_sentence_count)),
    )
    best_text = ""
    best_piece_index = 0
    best_score: tuple[int, ...] | None = None
    best_decision_rank = -1

    for piece_index, raw_piece in enumerate(pieces):
        piece = str(raw_piece or "").strip()
        if not piece:
            continue
        sentences = [
            re.sub(r"\s+", " ", part).strip()
            for part in re.split(r"(?<=[.!?。！？])(?:\s+|$)|[\r\n]+", piece)
            if part.strip()
        ] or [re.sub(r"\s+", " ", piece).strip()]
        sentence_sequences: list[tuple[int, list[str]]] = [
            (start, sentences[start : start + 3])
            for start in range(len(sentences))
        ]
        for start, sentence in enumerate(sentences):
            for offset in _focus_slice_offsets(sentence, content_focus):
                if offset >= citation_chars:
                    sentence_sequences.append((start, [sentence[offset:]]))

        for start, sequence in sentence_sequences:
            window_parts: list[str] = []
            for window_length, sentence in enumerate(sequence, start=1):
                candidate = " ".join((*window_parts, sentence)).strip()
                if window_parts and len(candidate) > max_chars:
                    break
                window_parts.append(sentence)
                window = " ".join(window_parts).strip()[:max_chars].strip()
                window = _trim_leading_code_fragment(window)
                if not window:
                    continue
                citation_prefix = window[: max(1, citation_chars)].strip()
                if (
                    _NONASSERTIVE_QUESTION.fullmatch(citation_prefix)
                    or "](#" in window_parts[0]
                    or all(
                        _NONASSERTIVE_HEADING.fullmatch(part)
                        for part in window_parts
                    )
                    and not (
                        purpose_question
                        and _PURPOSE_ANSWER.search(citation_prefix)
                    )
                ):
                    continue
                probe = EvidenceItem(
                    chunk_id=-1,
                    document_id=f"fetched-piece-{piece_index}-{start}-{window_length}",
                    url=document_url,
                    title=document_title,
                    text=citation_prefix,
                )
                outcome = assess_evidence(excerpt_intent, [probe]).outcomes[0]
                content_outcome = assess_evidence(
                    content_intent,
                    [
                        EvidenceItem(
                            chunk_id=-1,
                            document_id=f"fetched-text-{piece_index}-{start}-{window_length}",
                            url="",
                            title="",
                            text=citation_prefix,
                        )
                    ],
                ).outcomes[0]
                problem_solution_overview_signal = _problem_solution_overview_signal(
                    excerpt_intent.research_focus,
                    citation_prefix,
                )
                primary_cause_signal = bool(
                    _PRIMARY_CAUSE_QUERY.search(excerpt_intent.research_focus)
                    and _PRIMARY_CAUSE_ANSWER.search(citation_prefix)
                )
                purpose_signal = bool(
                    purpose_question and _PURPOSE_ANSWER.search(citation_prefix)
                )
                definition_signal = bool(
                    definition_question
                    and _definition_answer_signal(
                        excerpt_intent.research_focus,
                        citation_prefix,
                    )
                )
                definition_identity_signal = bool(
                    definition_question
                    and _definition_identity_signal(
                        excerpt_intent.research_focus,
                        citation_prefix,
                    )
                )
                overview_subject_rank = (
                    _overview_answer_rank(
                        excerpt_intent.research_focus,
                        citation_prefix,
                    )
                    if overview_question
                    else 0
                )
                overview_subject_signal = overview_subject_rank > 0
                query_syntax_rank = (
                    _query_syntax_answer_rank(citation_prefix)
                    if query_syntax_question
                    else 0
                )
                ranking_answer_rank = (
                    _ranking_answer_rank(citation_prefix)
                    if ranking_question
                    else 0
                )
                leading_prefix = window_parts[0][: max(1, citation_chars)].strip()
                leading_definition_subject = bool(
                    definition_question
                    and _definition_subject_leads_window(
                        excerpt_intent.research_focus,
                        leading_prefix,
                    )
                )
                leading_answer_signal = bool(
                    (
                        _PRIMARY_CAUSE_QUERY.search(excerpt_intent.research_focus)
                        and _PRIMARY_CAUSE_ANSWER.search(leading_prefix)
                    )
                    or (purpose_question and _PURPOSE_ANSWER.search(leading_prefix))
                    or (
                        definition_question
                        and _definition_answer_signal(
                            excerpt_intent.research_focus,
                            leading_prefix,
                        )
                    )
                    or (
                        _CAUSE_QUERY.search(excerpt_intent.research_focus)
                        and _CAUSE_ANSWER.search(leading_prefix)
                    )
                )
                if (
                    content_outcome.decision != "eligible"
                    and not problem_solution_overview_signal
                    and not primary_cause_signal
                    and not purpose_signal
                    and not definition_signal
                    and not query_syntax_rank
                    and not ranking_answer_rank
                ):
                    continue
                if purpose_question and not purpose_signal:
                    continue
                if definition_question and not definition_signal:
                    continue
                if overview_question and not overview_subject_signal:
                    continue
                word_count = len(_SNIPPET_TOKEN.findall(citation_prefix))
                assertive_sentence_count = sum(
                    not bool(_NONASSERTIVE_HEADING.fullmatch(part))
                    for part in window_parts
                )
                outcome_decision_rank = decision_rank[outcome.decision]
                score = (
                    int(primary_cause_signal),
                    int(purpose_signal),
                    int(definition_signal),
                    int(definition_identity_signal),
                    overview_subject_rank,
                    query_syntax_rank,
                    ranking_answer_rank,
                    int(leading_definition_subject),
                    int(leading_answer_signal),
                    int(
                        bool(_CAUSE_QUERY.search(excerpt_intent.research_focus))
                        and bool(_CAUSE_ANSWER.search(citation_prefix))
                    ),
                    int(
                        not semantic_binding_conflict(
                            excerpt_intent.research_focus,
                            citation_prefix,
                        )
                    ),
                    int(problem_solution_overview_signal),
                    decision_rank[content_outcome.decision],
                    relevance_rank[content_outcome.relevance],
                    len(content_outcome.matched_anchors),
                    outcome_decision_rank,
                    relevance_rank[outcome.relevance],
                    len(outcome.matched_anchors),
                    min(assertive_sentence_count, desired_sentence_count),
                    -word_count,
                    -piece_index,
                    -start,
                )
                if best_score is None or score > best_score:
                    best_score = score
                    best_text = window
                    best_piece_index = piece_index
                    best_decision_rank = outcome_decision_rank
                if len(candidate) >= max_chars:
                    break

    if best_score is None or best_decision_rank < decision_rank["eligible"]:
        return "", best_piece_index
    return best_text, best_piece_index


def _trim_leading_code_fragment(text: str) -> str:
    """Drop a dangling code prefix when the same excerpt contains prose after it."""

    stripped = text.strip()
    if "```" not in stripped:
        return stripped
    prefix, prose = stripped.split("```", 1)
    prose = prose.strip()
    if not prose:
        return stripped
    if not prefix.strip() or prefix.lstrip().startswith("...") or re.match(
        r"^\s*(?:MATCH|SELECT|INSERT|UPDATE|DELETE|CREATE|WITH|rank\s*=|rank\s+MATCH)\b",
        prefix,
        re.IGNORECASE,
    ):
        return prose
    return stripped


def _overview_answer_rank(focus: str, text: str) -> int:
    """Rank overview text by how directly it defines the distinctive subject."""

    if not _OVERVIEW_ANSWER.search(text):
        return 0
    tokens = [
        token.lower()
        for token in _SNIPPET_TOKEN.findall(focus)
        if len(token) > 2
        and token.lower() not in _FOCUS_EXCERPT_STOPWORDS
        and token.lower() not in _OVERVIEW_FOCUS_STOPWORDS
    ]
    if not tokens:
        return 1
    anchor = max(
        dict.fromkeys(tokens),
        key=lambda token: (
            int(any(char.isdigit() for char in token)),
            len(token),
            token,
        ),
    )
    text_tokens = set(_SNIPPET_TOKEN.findall(text.lower()))
    if anchor not in text_tokens:
        return 0
    if re.search(r"[\u3400-\u9fff]", anchor):
        return 2
    predicate = r"(?:is|are|supports?|provides?|offers?|enables?|allows?|features?|implements?|uses?)"
    for gap in range(4):
        if re.search(
            rf"\b{re.escape(anchor)}\b(?:\s+[a-z][a-z0-9'’-]*){{{gap}}}\s+{predicate}\b",
            text,
            re.IGNORECASE,
        ):
            return 4 - gap
    return 0


def _query_syntax_answer_rank(text: str) -> int:
    """Prefer concrete MATCH grammar over a generic syntax mention."""

    has_match = bool(re.search(r"\bMATCH\b", text, re.IGNORECASE))
    detail_categories = sum(
        (
            bool(re.search(r"\bphrases?\b|(?:短语)", text, re.IGNORECASE)),
            bool(re.search(r"\bprefix(?:es|\s+(?:tokens?|queries?))?\b|(?:前缀)", text, re.IGNORECASE)),
            bool(re.search(r"\bNEAR\b|(?:邻近)", text)),
            bool(re.search(r"\bboolean\b|(?:布尔)", text, re.IGNORECASE) or _QUERY_BOOLEAN_OPERATOR.search(text)),
            bool(re.search(r"\bcolumn\s+filters?\b|(?:列过滤)", text, re.IGNORECASE)),
            bool(re.search(r"\bbarewords?\b|double[-\s]?quot(?:e|ed)|(?:双引号)", text, re.IGNORECASE)),
        )
    )
    if detail_categories:
        rank = min(7, detail_categories + (2 if has_match else 1))
        if re.match(r"^[a-z]", text):
            rank = max(1, rank - 2)
        return rank
    if has_match and re.search(r"\b(?:query|syntax|expression)\b", text, re.IGNORECASE):
        return 1
    return 0


def _ranking_answer_rank(text: str) -> int:
    """Prefer explanatory rank/BM25 text over isolated SQL examples."""

    has_bm25 = bool(re.search(r"\bbm25\b", text, re.IGNORECASE))
    has_rank = bool(re.search(r"\b(?:rank|ranking|relevance)\b", text, re.IGNORECASE))
    has_explanation = bool(_RANKING_EXPLANATION.search(text))
    if has_bm25 and re.search(
        r"\bbm25(?:\(\))?\s+(?:auxiliary\s+)?function\b.{0,100}\breturns?\b|"
        r"\bbuilt-in\b.{0,60}\bbm25\b.{0,60}\breturns?\b",
        text,
        re.IGNORECASE,
    ):
        return 5
    if re.search(
        r"\brank\s+column\b.{0,120}\b(?:contains?|defaults?|same\s+value)\b",
        text,
        re.IGNORECASE,
    ):
        return 4
    if has_bm25 and re.search(
        r"\bbetter\s+matches?\b.{0,80}\b(?:smaller|lower)\b|"
        r"\b(?:smaller|lower)\b.{0,80}\bbetter\s+matches?\b",
        text,
        re.IGNORECASE,
    ):
        return 4
    if has_bm25 and has_explanation:
        return 3
    if has_rank and has_explanation:
        return 2
    if has_bm25 or has_rank:
        return 1
    return 0


def _definition_answer_signal(focus: str, text: str) -> bool:
    """Require an answer-bearing definition tied to the requested subject."""

    if not _DEFINITION_ANSWER.search(text):
        return False
    match = _DEFINITION_SUBJECT.search(focus.strip())
    if not match:
        return True
    subject = str(match.group("subject") or match.group("does_subject") or "").strip()
    subject_tokens = [
        token
        for token in _SNIPPET_TOKEN.findall(subject.lower())
        if len(token) > 2 and token not in _FOCUS_EXCERPT_STOPWORDS
    ]
    if not subject_tokens:
        return True
    text_tokens = set(_SNIPPET_TOKEN.findall(text.lower()))
    numeric_tokens = {token for token in subject_tokens if any(char.isdigit() for char in token)}
    if numeric_tokens:
        lexical_tokens = [token for token in subject_tokens if token not in numeric_tokens]
        subject_head = lexical_tokens[-1] if lexical_tokens else ""
        return numeric_tokens <= text_tokens and bool(subject_head and subject_head in text_tokens)
    if "." in subject:
        return subject_tokens[-1] in text_tokens
    return all(token in text_tokens for token in subject_tokens[:8])


def _definition_identity_signal(focus: str, text: str) -> bool:
    """Prefer an exact signature or subject identity over related-class mentions."""

    match = _DEFINITION_SUBJECT.search(focus.strip())
    if not match:
        return False
    subject = str(match.group("subject") or match.group("does_subject") or "").strip()
    dotted = re.search(r"([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+)", subject)
    if dotted:
        return bool(
            re.search(
                rf"(?<![A-Za-z0-9_]){re.escape(dotted.group(1))}\s*\(",
                text,
            )
        )
    subject_tokens = [
        token
        for token in _SNIPPET_TOKEN.findall(subject.lower())
        if len(token) > 2 and token not in _FOCUS_EXCERPT_STOPWORDS
    ]
    text_tokens = set(_SNIPPET_TOKEN.findall(text.lower()))
    numeric_tokens = {token for token in subject_tokens if any(char.isdigit() for char in token)}
    if numeric_tokens:
        lexical_tokens = [token for token in subject_tokens if token not in numeric_tokens]
        return numeric_tokens <= text_tokens and bool(
            lexical_tokens and lexical_tokens[-1] in text_tokens
        )
    return bool(subject_tokens) and all(token in text_tokens for token in subject_tokens[:8])


def _definition_subject_leads_window(focus: str, text: str) -> bool:
    """Prefer windows that introduce the requested definition subject before its predicate."""

    match = _DEFINITION_SUBJECT.search(focus.strip())
    if not match:
        return False
    subject = str(match.group("subject") or match.group("does_subject") or "").strip()
    dotted = re.search(r"\.([A-Za-z_][A-Za-z0-9_]*)", subject)
    if dotted:
        return bool(re.search(rf"(?<!\w){re.escape(dotted.group(1))}(?!\w)", text))
    subject_tokens = [
        token
        for token in _SNIPPET_TOKEN.findall(subject.lower())
        if len(token) > 2 and token not in _FOCUS_EXCERPT_STOPWORDS
    ]
    text_tokens = set(_SNIPPET_TOKEN.findall(text.lower()))
    numeric_tokens = {token for token in subject_tokens if any(char.isdigit() for char in token)}
    if numeric_tokens:
        return numeric_tokens <= text_tokens
    return bool(subject_tokens) and all(token in text_tokens for token in subject_tokens[:8])


def _problem_solution_overview_signal(focus: str, text: str) -> bool:
    """Reject negated limitations that merely resemble a product overview."""

    if not _PROBLEM_SOLUTION_QUERY.search(focus):
        return False
    for match in _PROBLEM_SOLUTION_OVERVIEW.finditer(text):
        matched_text = match.group(0)
        if _NONCONTIGUOUS_EXCERPT_GAP.search(matched_text):
            continue
        if not _NEGATED_PROBLEM_SOLUTION_ACTION.search(matched_text):
            return True
    return False


def _focus_slice_offsets(text: str, focus: str) -> tuple[int, ...]:
    """Return bounded subject/topic offsets for flattened tables or lists."""

    anchors: list[str] = []
    for raw in _SNIPPET_TOKEN.findall(str(focus or "").lower()):
        token = str(raw).strip("._+-")
        if len(token) < 3 or token in _FOCUS_EXCERPT_STOPWORDS:
            continue
        for variant in (token, token[:-1] if token.endswith("s") and len(token) > 4 else ""):
            if variant and variant not in anchors:
                anchors.append(variant)

    offsets: set[int] = set()
    for anchor in anchors[:12]:
        pattern = re.compile(rf"(?<!\w){re.escape(anchor)}(?!\w)", re.IGNORECASE)
        for match in pattern.finditer(text):
            token_start = text.rfind(" ", 0, match.start()) + 1
            token_prefix = text[token_start : match.start()].lower()
            if "://" in token_prefix or token_prefix.startswith("www."):
                continue
            if match.start() > 0:
                offsets.add(match.start())
            if len(offsets) >= 32:
                return tuple(sorted(offsets))
    return tuple(sorted(offsets))


def snippet_evidence(candidate: ExternalCandidate) -> EvidenceItem | None:
    """Materialize only a safe, direct snippet candidate as transient evidence."""

    if candidate.content_basis != "snippet" or candidate.prefetch_decision != "candidate":
        return None
    return EvidenceItem(
        chunk_id=-1,
        document_id=candidate.provenance_root,
        url=candidate.canonical_url,
        title=candidate.title,
        text=candidate.snippet,
        score=1.0 / (1 + candidate.rank),
        provenance_refs=[candidate.provenance_root],
        published_at=candidate.published_at,
        updated_at=candidate.updated_at,
        effective_at=candidate.effective_at,
        retrieved_at=candidate.retrieved_at,
        search_provider=candidate.provider,
        fetch_provider=candidate.fetch_provider,
    )


def _same_document_host(left: str, right: str) -> bool:
    def host(value: str) -> str:
        hostname = (urllib.parse.urlsplit(value).hostname or "").lower()
        return hostname[4:] if hostname.startswith("www.") else hostname

    return bool(host(left) and host(left) == host(right))


def _timestamp(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
