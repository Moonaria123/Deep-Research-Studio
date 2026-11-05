"""First-class citations and the GROUNDING_REQUIRED gate.

Every answer carries structured ``citations`` and a ``claim_map`` linking each
substantive claim to supporting citation ids drawn from the evidence pack. The
grounding gate enforces citation coverage strictly above 85% (baseline §验收指标).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .evidence_trust import EvidenceConflict, detect_temporal_intent
from .errors import BaselineError, ErrorCode
from .retrieval import EvidenceItem

COMPLEMENTARY_EVIDENCE_CHANNEL = "complementary_excerpt"
COMPLEMENTARY_CITATION_SNIPPET_CHARS = 4096

_LATIN_TOKEN = re.compile(r"[a-z0-9]+")
_LATIN_RAW_TOKEN = re.compile(r"\$?[A-Za-z0-9]+")
_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]+")
_INLINE_CITATION = re.compile(r"\[(c[0-9]+)\]", re.IGNORECASE)
_CITATION_ID = re.compile(r"c[0-9]+", re.IGNORECASE)
_LATIN_PROPER_TOKEN = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:[-.][A-Za-z0-9]+)*\b")
_NUMERIC_LITERAL = re.compile(
    r"(?<![A-Za-z0-9_])(?P<currency>[$€£¥])?"
    r"(?P<number>\d+(?:[.,]\d+)*)"
    r"(?:\s*(?P<unit>%|percent(?:age)?|basis\s+points?|bps|"
    r"milliseconds?|seconds?|minutes?|hours?|days?|weeks?|months?|years?|"
    r"bytes?|kilobytes?|megabytes?|gigabytes?|terabytes?|kb|mb|gb|tb|"
    r"millimet(?:er|re)s?|centimet(?:er|re)s?|kilomet(?:er|re)s?|"
    r"milligrams?|grams?|kilograms?|tonnes?|tons?|"
    r"usd|eur|gbp|cny|rmb|dollars?|euros?|pounds?|yuan|"
    r"thousand|million|billion|trillion))?(?!\d)",
    re.IGNORECASE,
)
_MARKDOWN_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+")
_BOOLEAN_OPERATOR_SEQUENCE = re.compile(
    r"\bAND\b\s*(?:,|/|\|)?\s*\bOR\b\s*(?:,|/|\|)?\s*(?:AND\s+)?\bNOT\b",
    re.IGNORECASE,
)
_TERMINAL_PUNCTUATION = frozenset(".!?;。！？；")
_CJK_NEGATION_MARKERS = ("并非", "不是", "不会", "不能", "没有", "未", "无", "不")
_CJK_NEGATION_EXEMPT_WORDS = ("未来", "无锡", "不丹")
_CJK_OPPOSITION_PAIRS = (
    ("提高", "降低"),
    ("增加", "减少"),
    ("上升", "下降"),
    ("增长", "下跌"),
    ("支持", "反对"),
    ("允许", "禁止"),
    ("成功", "失败"),
    ("改善", "恶化"),
    ("扩大", "缩小"),
    ("加快", "减缓"),
    ("提前", "推迟"),
    ("盈利", "亏损"),
    ("利好", "利空"),
)
_LATIN_NEGATION_TOKENS = frozenset({"no", "not", "never", "without", "cannot"})
_LATIN_NEGATION_CONTRACTION = re.compile(r"\b[a-z]+n['’]t\b", re.IGNORECASE)
_LATIN_NEGATION_EXPRESSION = re.compile(
    r"\b(?:no|not|never|without|cannot)\b|\b[a-z]+n['’]t\b",
    re.IGNORECASE,
)
_CAPABILITY_NOMINALIZATION = re.compile(
    r"(?<!lack )(?<!lacks )(?<!lacking )\b(?:the\s+)?(?:ability|capability)\s+to\b",
    re.IGNORECASE,
)
_LATIN_ENUMERATION = re.compile(
    r"\b([a-z][a-z-]{2,})\s*,\s*"
    r"([a-z][a-z-]{2,})\s*,\s*"
    r"(?:(?:and|or)\s+)?([a-z][a-z-]{2,})"
    r"(?:\s*,?\s*(?:and|or)\s+([a-z][a-z-]{2,}))?",
    re.IGNORECASE,
)
_LATIN_DIRECTION_MODIFIERS = frozenset(
    {
        "annual",
        "annually",
        "considerably",
        "materially",
        "notably",
        "sharply",
        "significantly",
        "slightly",
        "substantially",
        "yearly",
    }
)
_SHORT_ENTITY_PREFIXES = frozenset(
    {"class", "company", "group", "model", "product", "region", "segment", "team", "ticker", "unit", "variant"}
)
_CJK_DIRECTION_MODIFIERS = ("公司", "年度", "显著", "明显", "大幅", "小幅")
_DIRECTION_CLAUSE_SPLIT = re.compile(
    r"(?:[，,；;。！？!?]+|\b(?:while|whereas|but|however|and|or)\b|然而|同时|而|但|却)",
    re.IGNORECASE,
)
_LATIN_OPPOSITION_GROUPS = (
    (
        frozenset(
            {
                "increase",
                "increased",
                "increases",
                "increasing",
                "higher",
                "raise",
                "raised",
                "rise",
                "rises",
                "rose",
                "growth",
                "grow",
                "grew",
            }
        ),
        frozenset(
            {
                "decrease",
                "decreased",
                "decreases",
                "decreasing",
                "lower",
                "reduce",
                "reduced",
                "reduces",
                "decline",
                "declined",
                "fall",
                "fell",
                "drop",
                "dropped",
            }
        ),
    ),
    (
        frozenset(
            {
                "allow",
                "allowed",
                "allows",
                "available",
                "can",
                "enable",
                "enabled",
                "enables",
                "feature",
                "featured",
                "features",
                "include",
                "included",
                "includes",
                "including",
                "offer",
                "offered",
                "offers",
                "provide",
                "provided",
                "provides",
                "support",
                "supported",
                "supports",
            }
        ),
        frozenset(
            {
                "lack",
                "lacked",
                "lacking",
                "lacks",
                "oppose",
                "opposed",
                "opposes",
            }
        ),
    ),
    (
        frozenset(
            {
                "allow",
                "allows",
                "allowed",
                "execute",
                "executes",
                "executed",
                "execution",
            }
        ),
        frozenset(
            {
                "prohibit",
                "prohibits",
                "prohibited",
                "ban",
                "banned",
            }
        ),
    ),
    (
        frozenset({"success", "successful", "succeed", "succeeded"}),
        frozenset({"failure", "failed", "fail", "unsuccessful"}),
    ),
)
_STRUCTURAL_HEADING_EXACT = frozenset(
    {
        "abstract",
        "analysis",
        "appendix",
        "approach",
        "background",
        "caveats",
        "conclusion",
        "discussion",
        "evidence conflict note",
        "executive summary",
        "findings",
        "introduction",
        "key takeaways",
        "limitations",
        "methodology",
        "methods",
        "overview",
        "process",
        "recommendations",
        "references",
        "research process",
        "results",
        "scope",
        "sources",
        "summary",
        "temporal note",
    }
)
_TOPIC_HEADING_SUFFIX = re.compile(
    r"\b(?:analysis|applications?|architecture|background|capabilities|comparison|context|"
    r"design|dynamics|ecosystem|evidence|features|findings|implementation|landscape|"
    r"limitations|methodology|methods|modes|options|outlook|overview|performance|process|"
    r"recommendations|references|results|risks|security|semantics|sources|syntax|trends|"
    r"use cases|workflow)\s*$"
    r"|(?:概览|概述|背景|上下文|范围|方法|流程|工作流|架构|能力|功能|选项|模式|语义|语法|"
    r"设计|实现|分析|比较|对比|展望|趋势|格局|动态|生态|用例|应用|性能|安全|限制|风险|"
    r"建议|结论|证据|来源|参考|附录)\s*$",
    re.IGNORECASE,
)
_NONCURRENT_CAPABILITY_LATIN = re.compile(
    r"\b(?:coming|expected|future|plan|planned|planning|plans|roadmap|scheduled|will)\b",
    re.IGNORECASE,
)
_NONCURRENT_CAPABILITY_CJK = re.compile(r"(?:即将|将在|未来|计划|规划|路线图|尚未)")
_SOURCE_REFERENCE_PROVIDED = re.compile(
    r"\b(?:(?:source|citation|evidence|document|documentation|entry|excerpt|"
    r"glossary(?:\s+entry)?|materials?|passage|reference|text)\s+provided|"
    r"provided\s+(?:source|citation|evidence|document|documentation|entry|excerpt|"
    r"glossary(?:\s+entry)?|materials?|passage|reference|text)|"
    r"(?:the|a|an)\s+provided\s+(?:[a-z0-9]+(?:-[a-z0-9]+)*\s+){1,4}"
    r"(?:source|citation|evidence|document|documentation|entry|excerpt|"
    r"glossary(?:\s+entry)?|materials?|passage|reference|example|text))\b",
    re.IGNORECASE,
)
_COMMON_ABBREVIATIONS = (
    "e.g.", "i.e.", "mr.", "mrs.", "ms.", "dr.", "prof.",
    "sr.", "jr.", "vs.", "etc.", "u.s.", "u.k.",
)
_STOPWORDS = frozenset(
    "the a an and or of to in for on with is are was were be by as at from this that these those it its "
    "we you they he she i our your their".split()
)
_LATIN_TOKEN_EQUIVALENTS = {
    "annual": "year",
    "annually": "year",
    "yearly": "year",
    "produced": "produce",
    "produces": "produce",
    "producing": "produce",
    "production": "produce",
    "execution": "execute",
}

_NON_ENTITY_PROPER_TOKENS = frozenset(
    {
        "A",
        "According",
        "An",
        "As",
        "At",
        "Based",
        "By",
        "For",
        "From",
        "If",
        "In",
        "It",
        "Its",
        "On",
        "The",
        "Their",
        "These",
        "This",
        "Those",
        "To",
        "Using",
        "We",
        "When",
        "While",
        "With",
        "You",
    }
)


@dataclass(frozen=True)
class Citation:
    """A citation referencing an evidence-pack item."""

    id: str
    document_id: str
    url: str
    title: str
    snippet: str
    score: float
    origin: str = "external"
    artifact_type: str = "web_document"
    artifact_id: str = ""
    grounding_eligible: bool = True
    provenance_refs: tuple[str, ...] | list[str] = field(default_factory=tuple)
    source_status: str = "active"
    published_at: float | None = None
    updated_at: float | None = None
    effective_at: float | None = None
    retrieved_at: float | None = None
    search_provider: str = ""
    fetch_provider: str = ""

    def to_dict(self) -> dict:
        """Return a JSON-serializable view."""

        return {
            "id": self.id,
            "document_id": self.document_id,
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
            "score": round(self.score, 6),
            "origin": self.origin,
            "artifact_type": self.artifact_type,
            "artifact_id": self.artifact_id,
            "grounding_eligible": self.grounding_eligible,
            "provenance_refs": list(self.provenance_refs),
            "source_status": self.source_status,
            "published_at": self.published_at,
            "updated_at": self.updated_at,
            "effective_at": self.effective_at,
            "retrieved_at": self.retrieved_at,
            "search_provider": self.search_provider or None,
            "fetch_provider": self.fetch_provider or None,
        }


@dataclass
class GroundedAnswer:
    """An answer with first-class citations and a claim map.

    ``text`` is the canonical Markdown body and is kept confident and declarative.
    ``summary`` (executive summary) and ``key_takeaways`` are optional structured
    views derived from ``text``; ``limitations`` holds caveats/unresolved-conflict
    notes that are split OUT of the body so the report does not mix alarming
    hedges into its findings. ``confidence_label`` is a calm qualitative grading
    (``established``/``contested``/``uncertain``) shown instead of a bare percentage.
    None of these introduce ungrounded content.
    """

    text: str
    citations: list[Citation] = field(default_factory=list)
    claim_map: dict[str, list[str]] = field(default_factory=dict)
    coverage: float = 0.0
    summary: str = ""
    key_takeaways: list[str] = field(default_factory=list)
    limitations: str = ""
    confidence_label: str = ""
    conflicts: list[EvidenceConflict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-serializable view."""

        return {
            "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "claim_map": self.claim_map,
            "coverage": round(self.coverage, 4),
            "summary": self.summary,
            "key_takeaways": list(self.key_takeaways),
            "limitations": self.limitations,
            "confidence_label": self.confidence_label,
            "conflicts": [conflict.to_dict() for conflict in self.conflicts],
        }


@dataclass(frozen=True)
class ClaimGroundingDiagnostic:
    """Read-only verification result for one user-visible substantive claim."""

    claim: str
    citation_ids: tuple[str, ...]
    supported: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "claim": self.claim,
            "citation_ids": list(self.citation_ids),
            "supported": self.supported,
        }


_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$")
_HEADING_LEVEL = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
_BULLET = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.*\S)\s*$")
_SUMMARY_ALIASES = ("executive summary", "summary", "overview", "tl;dr", "abstract")
_TAKEAWAY_ALIASES = ("key takeaways", "takeaways", "key points", "key findings", "highlights")
# Limitations / caveats / open-questions section titles (English + Chinese) that
# should be split OUT of the confident report body and shown in a calm panel.
_LIMITATION_ALIASES = (
    "limitations and open questions",
    "limitations & open questions",
    "limitations",
    "limitation",
    "caveats and open questions",
    "caveats",
    "caveat",
    "open questions",
    "局限性与客观批评",
    "局限性与未决问题",
    "局限与未决问题",
    "局限性",
    "局限",
    "未决问题",
    "不确定性",
    "不确定",
)

# Confidence label vocabulary (calm, qualitative) shown instead of a raw percentage.
CONFIDENCE_ESTABLISHED = "established"
CONFIDENCE_CONTESTED = "contested"
CONFIDENCE_UNCERTAIN = "uncertain"


def _heading_text(line: str) -> str | None:
    match = _HEADING.match(line)
    return match.group(1).strip().lower() if match else None


def _first_paragraph(lines: list[str]) -> str:
    buffer: list[str] = []
    for line in lines:
        if _heading_text(line) is not None or _BULLET.match(line):
            if buffer:
                break
            continue
        if not line.strip():
            if buffer:
                break
            continue
        buffer.append(line.strip())
    return " ".join(buffer).strip()


def extract_report_structure(text: str) -> tuple[str, list[str]]:
    """Parse a Markdown report into (executive summary, key takeaways).

    Looks for headed ``Executive Summary``/``Key Takeaways`` sections; falls back to
    the first prose paragraph for the summary when no summary heading is present.
    """

    lines = text.splitlines()
    summary_lines: list[str] = []
    takeaways: list[str] = []
    current: str | None = None
    for line in lines:
        heading = _heading_text(line)
        if heading is not None:
            if any(heading == alias or heading.startswith(alias) for alias in _SUMMARY_ALIASES):
                current = "summary"
            elif any(heading == alias or heading.startswith(alias) for alias in _TAKEAWAY_ALIASES):
                current = "takeaways"
            else:
                current = None
            continue
        if current == "summary":
            if line.strip():
                summary_lines.append(line.strip())
            elif summary_lines:
                current = None
        elif current == "takeaways":
            bullet = _BULLET.match(line)
            if bullet:
                takeaways.append(bullet.group(1).strip())
    summary = " ".join(summary_lines).strip() or _first_paragraph(lines)
    return summary, takeaways


def _is_limitations_heading(title: str) -> bool:
    cleaned = title.strip().lower()
    return any(cleaned == alias or cleaned.startswith(alias) for alias in _LIMITATION_ALIASES)


def split_limitations(text: str) -> tuple[str, str]:
    """Split a Markdown report into ``(body_without_limitations, limitations_text)``.

    Finds the first heading whose title matches a limitations alias and removes that
    section (down to the next heading of the same or higher level) from the body,
    returning its inner text separately. When no such section exists the original
    text is returned unchanged with an empty limitations string, so the operation is
    idempotent and safe to apply more than once.
    """

    lines = text.splitlines()
    start: int | None = None
    level = 0
    for index, line in enumerate(lines):
        match = _HEADING_LEVEL.match(line)
        if match and _is_limitations_heading(match.group(2)):
            start = index
            level = len(match.group(1))
            break
    if start is None:
        return text, ""

    end = len(lines)
    for index in range(start + 1, len(lines)):
        match = _HEADING_LEVEL.match(lines[index])
        if match and len(match.group(1)) <= level:
            end = index
            break

    limitations = "\n".join(lines[start + 1 : end]).strip()
    body = "\n".join(lines[:start] + lines[end:]).strip()
    return body, limitations


def derive_confidence_label(coverage: float, agreement: float | None = None) -> str:
    """Map grounding coverage (and optional council agreement) to a calm label.

    Returns one of ``established`` (well supported), ``contested`` (mixed support or
    models diverge), or ``uncertain`` (lightly grounded). The label is presentational
    only -- it replaces an alarming bare percentage with an intuitive grade. Council
    callers pass ``agreement`` so low cross-model agreement surfaces as ``contested``
    rather than implying the finding itself is weak.
    """

    if coverage < 0.55:
        return CONFIDENCE_UNCERTAIN
    if agreement is not None and agreement < 0.50:
        return CONFIDENCE_CONTESTED
    if agreement is None and coverage < 0.80:
        return CONFIDENCE_UNCERTAIN
    return CONFIDENCE_ESTABLISHED


def enrich_structure(answer: GroundedAnswer) -> GroundedAnswer:
    """Shape an answer for confident rendering.

    Populates ``summary``/``key_takeaways`` from the Markdown body, splits any
    Limitations section out into ``limitations`` (removing it from ``text`` so the
    body reads as a confident briefing), and derives a coverage-based
    ``confidence_label`` when one is not already set (council overrides it later).
    """

    summary, takeaways = extract_report_structure(answer.text)
    body, limitations = split_limitations(answer.text)
    answer.text = body
    answer.summary = summary
    answer.key_takeaways = takeaways
    answer.limitations = limitations
    if not answer.confidence_label:
        answer.confidence_label = derive_confidence_label(answer.coverage)
    return answer


def enrich_council_structure(answer: GroundedAnswer) -> GroundedAnswer:
    """Shape a council consensus for confident rendering.

    Unlike :func:`enrich_structure`, this never extracts ``summary``/``key_takeaways``:
    the council consensus is already structured around Agreement/Disagreement/Synthesis,
    so deriving an Executive Summary from those headings would mis-label an Agreement
    bullet as the headline finding. It only splits any Limitations section out of the
    body and derives a coverage-based ``confidence_label`` when one is not already set
    (``run_council`` sets an agreement-aware label, which is preserved).
    """

    body, limitations = split_limitations(answer.text)
    answer.text = body
    answer.limitations = limitations
    if not answer.confidence_label:
        answer.confidence_label = derive_confidence_label(answer.coverage)
    return answer


def build_citations(evidence: list[EvidenceItem], *, snippet_chars: int = 240) -> list[Citation]:
    """Convert evidence items into stable citations (``c1``..``cN``)."""

    citations: list[Citation] = []
    eligible = [item for item in evidence if item.grounding_eligible]
    for index, item in enumerate(eligible, start=1):
        item_snippet_chars = (
            max(snippet_chars, COMPLEMENTARY_CITATION_SNIPPET_CHARS)
            if COMPLEMENTARY_EVIDENCE_CHANNEL in item.channels
            else snippet_chars
        )
        citations.append(
            Citation(
                id=f"c{index}",
                document_id=item.document_id,
                url=item.url,
                title=item.title or item.url,
                snippet=item.text[:item_snippet_chars].strip(),
                score=item.score,
                origin=item.origin,
                artifact_type=item.artifact_type,
                artifact_id=item.artifact_id,
                grounding_eligible=item.grounding_eligible,
                provenance_refs=tuple(item.provenance_refs),
                source_status=item.source_status,
                published_at=item.published_at,
                updated_at=item.updated_at,
                effective_at=item.effective_at,
                retrieved_at=item.retrieved_at,
                search_provider=item.search_provider,
                fetch_provider=item.fetch_provider,
            )
        )
    return citations


def _latin_tokens(text: str) -> set[str]:
    return {
        _LATIN_TOKEN_EQUIVALENTS.get(token, token)
        for token in _LATIN_TOKEN.findall(text.lower())
        if token not in _STOPWORDS and len(token) > 2
    }


def _cjk_bigrams(text: str) -> set[str]:
    return {
        run[index : index + 2]
        for run in _CJK_RUN.findall(text)
        for index in range(len(run) - 1)
    }


def _cjk_text(text: str) -> str:
    return "".join(_CJK_RUN.findall(text))


def _common_prefix_length(left: str, right: str) -> int:
    index = 0
    while index < min(len(left), len(right)) and left[index] == right[index]:
        index += 1
    return index


def _directional_clauses(text: str) -> list[str]:
    return [part.strip() for part in _DIRECTION_CLAUSE_SPLIT.split(text) if part.strip()]


def _latin_direction_anchors(
    clause: str, positive: frozenset[str], negative: frozenset[str]
) -> set[str]:
    anchors: set[str] = set()
    raw_tokens = _LATIN_RAW_TOKEN.findall(clause)
    for index, raw_token in enumerate(raw_tokens):
        identifier = raw_token.lstrip("$")
        token = raw_token.lower() if raw_token.startswith("$") else identifier.lower()
        previous = raw_tokens[index - 1].lstrip("$").lower() if index else ""
        short_entity = bool(identifier) and (
            raw_token.startswith("$")
            or (
                identifier.isupper()
                and len(identifier) <= 2
                and identifier != "I"
            )
            or (len(identifier) == 1 and previous in _SHORT_ENTITY_PREFIXES)
        )
        if (
            token in positive
            or token in negative
            or token in _LATIN_DIRECTION_MODIFIERS
            or (token in _STOPWORDS and not short_entity)
        ):
            continue
        if len(token) > 2 or short_entity:
            anchors.add(token if short_entity else _latin_anchor_token(token))
    return anchors


def _latin_anchor_token(token: str) -> str:
    """Normalize conservative noun inflections used to bind a direction to its subject."""

    normalized = _LATIN_TOKEN_EQUIVALENTS.get(token, token)
    if len(normalized) > 4 and normalized.endswith("ies"):
        return f"{normalized[:-3]}y"
    if (
        len(normalized) > 4
        and normalized.endswith("s")
        and not normalized.endswith(("ss", "us", "is"))
    ):
        return normalized[:-1]
    return normalized


def _cjk_direction_anchors(clause: str, left: str, right: str) -> set[str]:
    cleaned = clause.replace(left, "").replace(right, "")
    for modifier in _CJK_DIRECTION_MODIFIERS:
        cleaned = cleaned.replace(modifier, "")
    compact = _cjk_text(cleaned)
    return _cjk_bigrams(compact) or ({compact} if compact else set())


def _direction_anchors_compatible(
    namespace: str,
    group: int,
    required: set[str],
    candidate: set[str],
) -> bool:
    if not candidate:
        return False
    if required <= candidate or candidate <= required:
        return True
    return namespace == "latin" and group == 1 and len(required & candidate) >= 2


def _directional_requirements(text: str) -> list[tuple[str, int, int, set[str], bool]]:
    requirements: list[tuple[str, int, int, set[str], bool]] = []
    normalized_text = _normalize_directional_text(text)
    for clause in _directional_clauses(normalized_text):
        tokens = set(_LATIN_TOKEN.findall(clause.lower()))
        for index, (positive, negative) in enumerate(_LATIN_OPPOSITION_GROUPS):
            direction = (bool(tokens & positive), bool(tokens & negative))
            if direction in {(True, False), (False, True)}:
                requirements.append(
                    (
                        "latin",
                        index,
                        1 if direction[0] else -1,
                        _latin_direction_anchors(clause, positive, negative),
                        index == 1 and bool(_NONCURRENT_CAPABILITY_LATIN.search(clause)),
                    )
                )
        for index, (left, right) in enumerate(_CJK_OPPOSITION_PAIRS):
            direction = (left in clause, right in clause)
            if direction in {(True, False), (False, True)}:
                requirements.append(
                    (
                        "cjk",
                        index,
                        1 if direction[0] else -1,
                        _cjk_direction_anchors(clause, left, right),
                        index == 4 and bool(_NONCURRENT_CAPABILITY_CJK.search(clause)),
                    )
                )
    return requirements


def _without_source_reference_provided(text: str) -> str:
    """Keep source-attribution wording from masquerading as a capability claim."""

    return _SOURCE_REFERENCE_PROVIDED.sub(
        lambda match: re.sub(r"\bprovided\b", "", match.group(0), flags=re.IGNORECASE),
        text,
    )


def _normalize_directional_text(text: str) -> str:
    """Normalize affirmative capability nouns without hiding explicit absence."""

    cleaned = _without_source_reference_provided(text)
    return _CAPABILITY_NOMINALIZATION.sub("can", cleaned)


def _directional_binding_conflict(claim: str, citation_text: str) -> bool:
    claim_requirements = _directional_requirements(claim)
    if not claim_requirements:
        return False
    citation_requirements = _directional_requirements(citation_text)
    for namespace, group, direction, anchors, noncurrent in claim_requirements:
        candidates = [
            (candidate_direction, candidate_anchors, candidate_noncurrent)
            for candidate_namespace, candidate_group, candidate_direction, candidate_anchors, candidate_noncurrent in citation_requirements
            if candidate_namespace == namespace and candidate_group == group
        ]
        matches = [
            candidate_anchors
            for candidate_direction, candidate_anchors, candidate_noncurrent in candidates
            if candidate_direction == direction and candidate_noncurrent == noncurrent
        ]
        if not matches or (
            anchors
            and not any(
                _direction_anchors_compatible(namespace, group, anchors, candidate)
                for candidate in matches
            )
        ):
            return True
    return False


def _has_cjk_negation(text: str) -> bool:
    cleaned = text
    for word in _CJK_NEGATION_EXEMPT_WORDS:
        cleaned = cleaned.replace(word, "")
    return any(marker in cleaned for marker in _CJK_NEGATION_MARKERS)


def _latin_predicate_stem(token: str) -> str:
    if token.endswith("es") and token[:-1].endswith("e"):
        return token[:-1]
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) > len(suffix) + 2:
            return token[: -len(suffix)]
    return token


def _latin_negated_predicates(text: str) -> set[str]:
    predicates: set[str] = set()
    directional_tokens = frozenset(
        token
        for positive, negative in _LATIN_OPPOSITION_GROUPS
        for token in (*positive, *negative)
    )
    for match in _LATIN_NEGATION_EXPRESSION.finditer(text):
        tail = re.split(r"[.!?;。！？；]", text[match.end() :], maxsplit=1)[0]
        tail = _DIRECTION_CLAUSE_SPLIT.split(tail, maxsplit=1)[0]
        candidates = [
            token
            for token in _LATIN_TOKEN.findall(tail.lower())
            if token not in _STOPWORDS and len(token) > 2
        ]
        if not candidates:
            continue
        predicate = next(
            (token for token in candidates if token in directional_tokens),
            candidates[0],
        )
        predicates.add(_latin_predicate_stem(predicate))
    return predicates


def _latin_negation_mismatch(left: str, right: str) -> bool:
    left_tokens = set(_LATIN_TOKEN.findall(left.lower()))
    right_tokens = set(_LATIN_TOKEN.findall(right.lower()))
    left_explicit = bool(left_tokens & _LATIN_NEGATION_TOKENS)
    right_explicit = bool(right_tokens & _LATIN_NEGATION_TOKENS)
    left_contracted = bool(_LATIN_NEGATION_CONTRACTION.search(left))
    right_contracted = bool(_LATIN_NEGATION_CONTRACTION.search(right))
    left_negated = left_explicit or left_contracted
    right_negated = right_explicit or right_contracted
    if left_negated == right_negated:
        return False

    negated_text = left if left_negated else right
    affirmative_text = right if left_negated else left
    predicates = _latin_negated_predicates(negated_text)
    if not predicates:
        return True
    affirmative_tokens = {
        _latin_predicate_stem(token) for token in _latin_tokens(affirmative_text)
    }
    return not predicates.isdisjoint(affirmative_tokens)


def _latin_enumeration_omission(claim: str, citation_text: str) -> bool:
    """Reject multi-item claim lists when the cited passage omits a listed term."""

    evidence_tokens = {
        _latin_predicate_stem(token)
        for token in _latin_tokens(citation_text)
    }
    for match in _LATIN_ENUMERATION.finditer(claim):
        required = {
            _latin_predicate_stem(token.lower())
            for token in match.groups()
            if token
        }
        if len(required) >= 3 and not required <= evidence_tokens:
            return True
    return False


def _without_inline_citations(text: str) -> str:
    return _INLINE_CITATION.sub("", text)


def _numeric_facts(text: str) -> set[tuple[str, str, str]]:
    """Return exact numeric facts without treating citation ids as numbers."""

    facts: set[tuple[str, str, str]] = set()
    for match in _NUMERIC_LITERAL.finditer(_without_inline_citations(text)):
        number = match.group("number").replace(",", "").lower()
        unit = re.sub(r"\s+", " ", (match.group("unit") or "").lower())
        currency = match.group("currency") or ""
        facts.add((currency, number, unit))
    return facts


def _numeric_fact_conflict(claim: str, citation_text: str) -> bool:
    """Reject changed numbers, dates, currency, scales, and explicit units."""

    claim_facts = _numeric_facts(claim)
    if not claim_facts:
        return False
    evidence_facts = _numeric_facts(citation_text)
    for currency, number, unit in claim_facts:
        if unit or currency:
            if (currency, number, unit) not in evidence_facts:
                return True
        elif not any(candidate_number == number for _, candidate_number, _ in evidence_facts):
            return True
    return False


def _capability_signatures(text: str) -> list[tuple[int, set[str], set[str], bool]]:
    """Return subject/object signatures for explicit Latin capability clauses."""

    positive, negative = _LATIN_OPPOSITION_GROUPS[1]
    signatures: list[tuple[int, set[str], set[str], bool]] = []
    clauses = re.split(r"[，,；;。！？!?]+", _normalize_directional_text(text))
    for clause in (part.strip() for part in clauses if part.strip()):
        raw_tokens = _LATIN_RAW_TOKEN.findall(clause)
        lowered = [token.lstrip("$").lower() for token in raw_tokens]
        indexes = [
            index
            for index, token in enumerate(lowered)
            if token in positive or token in negative
        ]
        for position, index in enumerate(indexes):
            direction = 1 if lowered[index] in positive else -1
            subject = {
                token.lower()
                for token in raw_tokens[:index]
                if _LATIN_PROPER_TOKEN.fullmatch(token)
                and token not in _NON_ENTITY_PROPER_TOKENS
            }
            object_end = indexes[position + 1] if position + 1 < len(indexes) else len(raw_tokens)
            object_text = " ".join(raw_tokens[index + 1 : object_end])
            objects = _latin_direction_anchors(object_text, positive, negative)
            signatures.append(
                (
                    direction,
                    subject,
                    objects,
                    bool(_NONCURRENT_CAPABILITY_LATIN.search(clause)),
                )
            )
    return signatures


def _capability_binding_conflict(claim: str, citation_text: str) -> bool:
    """Bind capability direction to the same named subject and object phrase."""

    claim_signatures = _capability_signatures(claim)
    if not claim_signatures:
        return False
    citation_signatures = _capability_signatures(citation_text)
    for direction, subject, objects, noncurrent in claim_signatures:
        # Absence, roadmap, and future statements are already compared by the
        # directional and negation checks. Treating ordinary verbs such as
        # "provide" as an entity capability here creates false subject/object
        # conflicts in sentences like "GDP does not provide...".
        if noncurrent:
            continue
        matched = False
        for candidate_direction, candidate_subject, candidate_objects, candidate_noncurrent in citation_signatures:
            if candidate_direction != direction or candidate_noncurrent != noncurrent:
                continue
            subject_compatible = (
                not subject
                or not candidate_subject
                or subject <= candidate_subject
                or candidate_subject <= subject
            )
            object_compatible = (
                not objects
                or objects <= candidate_objects
                or candidate_objects <= objects
                or len(objects & candidate_objects) >= 2
            )
            if subject_compatible and object_compatible:
                matched = True
                break
        if not matched:
            return True
    return False


def _has_semantic_conflict(claim: str, citation_text: str) -> bool:
    if _numeric_fact_conflict(claim, citation_text):
        return True
    if _capability_binding_conflict(claim, citation_text):
        return True
    if _directional_binding_conflict(claim, citation_text):
        return True
    if _latin_enumeration_omission(claim, citation_text):
        return True
    claim_lower = claim.lower()
    citation_lower = citation_text.lower()
    claim_tokens = set(_LATIN_TOKEN.findall(_normalize_directional_text(claim_lower)))
    citation_tokens = set(_LATIN_TOKEN.findall(_normalize_directional_text(citation_lower)))
    claim_natural_language = _BOOLEAN_OPERATOR_SEQUENCE.sub("", claim.replace("`", ""))
    citation_natural_language = _BOOLEAN_OPERATOR_SEQUENCE.sub("", citation_text.replace("`", ""))
    if _latin_negation_mismatch(claim_natural_language, citation_natural_language):
        return True
    for group, (positive, negative) in enumerate(_LATIN_OPPOSITION_GROUPS):
        claim_direction = (bool(claim_tokens & positive), bool(claim_tokens & negative))
        citation_direction = (
            bool(citation_tokens & positive),
            bool(citation_tokens & negative),
        )
        if (claim_direction[0] and not citation_direction[0]) or (
            claim_direction[1] and not citation_direction[1]
        ):
            return True
    claim_negated = _has_cjk_negation(claim)
    citation_negated = _has_cjk_negation(citation_text)
    if claim_negated != citation_negated:
        return True
    for group, (left, right) in enumerate(_CJK_OPPOSITION_PAIRS):
        claim_direction = (left in claim, right in claim)
        citation_direction = (left in citation_text, right in citation_text)
        if (claim_direction[0] and not citation_direction[0]) or (
            claim_direction[1] and not citation_direction[1]
        ):
            return True
    return False


def semantic_binding_conflict(claim: str, citation_text: str) -> bool:
    """Expose the directional claim-to-evidence binding check to retrieval."""

    return _has_semantic_conflict(claim, citation_text)


def _has_evidence_pair_conflict(left: str, right: str) -> bool:
    """Return true only when two evidence snippets contradict each other.

    ``_has_semantic_conflict`` is intentionally directional for grounding: a
    citation also fails when it merely omits a direction required by a claim.
    Evidence-pack conflict discovery needs an explicit opposite predicate, not
    two mutually incomplete descriptions. This keeps complementary definition
    snippets out of the bounded LLM conflict classifier.
    """

    left_requirements = _directional_requirements(left)
    right_requirements = _directional_requirements(right)
    for namespace, group, direction, anchors, noncurrent in left_requirements:
        for (
            candidate_namespace,
            candidate_group,
            candidate_direction,
            candidate_anchors,
            candidate_noncurrent,
        ) in right_requirements:
            if (
                candidate_namespace == namespace
                and candidate_group == group
                and candidate_direction == -direction
                and candidate_noncurrent == noncurrent
                and _direction_anchors_compatible(
                    namespace,
                    group,
                    anchors,
                    candidate_anchors,
                )
            ):
                return True

    left_negated = _latin_negated_predicates(left)
    right_negated = _latin_negated_predicates(right)
    if _has_asserted_latin_predicate(right, left_negated - right_negated):
        return True
    if _has_asserted_latin_predicate(left, right_negated - left_negated):
        return True

    # Preserve the existing CJK polarity contract until predicate-scoped CJK
    # negation extraction is available. Latin evidence uses the stricter rules
    # above so scoped prose such as "how to crawl" is not treated as an
    # affirmative opposite of "do not crawl certain paths".
    if _CJK_RUN.search(left) or _CJK_RUN.search(right):
        return semantic_binding_conflict(left, right) and semantic_binding_conflict(right, left)
    return False


def _has_asserted_latin_predicate(text: str, predicates: set[str]) -> bool:
    """Return whether text directly affirms one of the requested predicates."""

    if not predicates:
        return False
    negated = _latin_negated_predicates(text)
    for match in _LATIN_TOKEN.finditer(text):
        predicate = _latin_predicate_stem(match.group(0).lower())
        if predicate not in predicates or predicate in negated:
            continue
        prefix = text[max(0, match.start() - 80) : match.start()].lower()
        if re.search(r"\b(?:how|whether)\s+to\s+$", prefix):
            continue
        if re.search(
            r"\b(?:instruct|instructs|instructed|instructing|tell|tells|told|guide|guides|guided)"
            r"(?:\s+[a-z][a-z-]*){0,6}\s+to\s+$",
            prefix,
        ):
            continue
        return True
    return False


def _tokens(text: str) -> set[str]:
    return _latin_tokens(text) | _cjk_bigrams(text)


def _is_substantive(claim: str) -> bool:
    heading = _MARKDOWN_HEADING.match(claim)
    visible = _MARKDOWN_HEADING.sub("", claim, count=1).strip() if heading else claim
    if heading and not _inline_citation_ids(visible) and _is_structural_heading(visible):
        return False
    cjk_chars = sum(len(run) for run in _CJK_RUN.findall(visible))
    return len(_latin_tokens(visible)) >= 3 or cjk_chars >= 4


def _is_structural_heading(text: str) -> bool:
    visible = re.sub(r"^\s*(?:\d+(?:\.\d+)*[.)]?|[IVXLC]+[.)])\s+", "", text).strip()
    if visible.endswith(("?", "？")):
        return True
    normalized = re.sub(r"\s+", " ", visible).strip(" :-–—：").lower()
    return normalized in _STRUCTURAL_HEADING_EXACT or bool(_TOPIC_HEADING_SUFFIX.search(normalized))


def _has_sufficient_cjk_support(claim: str, citation_text: str) -> bool:
    """Require predicate-level CJK overlap and matching negation polarity."""

    claim_text = _cjk_text(claim)
    citation_cjk = _cjk_text(citation_text)
    claim_bigrams = _cjk_bigrams(claim)
    if not claim_bigrams:
        return True
    if claim_text in citation_cjk:
        return True
    shared = claim_bigrams & _cjk_bigrams(citation_text)
    if len(shared) < 2 or len(shared) / len(claim_bigrams) < 0.65:
        return False
    prefix_length = _common_prefix_length(claim_text, citation_cjk)
    if 4 <= prefix_length < len(claim_text):
        claim_tail = claim_text[prefix_length:]
        citation_tail = citation_cjk[prefix_length:]
        if len(claim_tail) >= 3:
            tail_bigrams = _cjk_bigrams(claim_tail)
            tail_shared = tail_bigrams & _cjk_bigrams(citation_tail)
            if len(tail_shared) < 2 or len(tail_shared) / max(1, len(tail_bigrams)) < 0.5:
                return False
    return True


def _unknown_citation_ids(answer: GroundedAnswer) -> set[str]:
    known = {citation.id.lower() for citation in answer.citations}
    inline = _inline_citation_ids(_grounding_text(answer))
    mapped = {
        str(citation_id).lower()
        for ids in answer.claim_map.values()
        for citation_id in ids
    }
    return (inline | mapped) - known


def _grounding_text(answer: GroundedAnswer) -> str:
    """Return every user-visible answer segment covered by the grounding gate."""

    if not answer.limitations.strip():
        return answer.text
    return f"{answer.text.rstrip()}\n{answer.limitations.strip()}".strip()


def _inline_citation_ids(text: str) -> set[str]:
    return {match.lower() for match in _INLINE_CITATION.findall(text)}


def _normalized_claim_map(
    claim_map: dict[str, list[str]],
) -> dict[str, set[str]] | None:
    normalized: dict[str, set[str]] = {}
    for claim, ids in claim_map.items():
        if not isinstance(claim, str) or not claim.strip() or not isinstance(ids, list) or not ids:
            return None
        if any(not isinstance(citation_id, str) for citation_id in ids):
            return None
        normalized_ids = [citation_id.lower() for citation_id in ids]
        if (
            len(set(normalized_ids)) != len(normalized_ids)
            or not all(_CITATION_ID.fullmatch(citation_id) for citation_id in normalized_ids)
        ):
            return None
        normalized[claim] = set(normalized_ids)
    return normalized


def _citation_bindings_valid(
    answer: GroundedAnswer, *, require_complete_map: bool = False
) -> bool:
    grounding_text = _grounding_text(answer)
    inline_ids = _inline_citation_ids(grounding_text)
    substantive_claims = [
        claim
        for claim in _split_claims(grounding_text)
        if _is_substantive(claim)
    ]
    if not substantive_claims:
        return not inline_ids and not answer.claim_map
    producer_map = _normalized_claim_map(answer.claim_map)
    if producer_map is None or not producer_map:
        return False

    recomputed = ground_answer(grounding_text, [], citations=answer.citations)
    recomputed_map = _normalized_claim_map(recomputed.claim_map)
    if recomputed_map is None:
        return False

    expected_claims = set(substantive_claims)
    if set(producer_map) != expected_claims or set(recomputed_map) != expected_claims:
        return False
    for claim in substantive_claims:
        claim_inline = _inline_citation_ids(claim)
        if claim_inline and (
            producer_map.get(claim) != claim_inline
            or recomputed_map.get(claim) != claim_inline
        ):
            return False
    if require_complete_map and inline_ids != {
        citation_id
        for claim in substantive_claims
        for citation_id in _inline_citation_ids(claim)
    }:
        return False
    return producer_map == recomputed_map


def claim_grounding_diagnostics(answer: GroundedAnswer) -> list[ClaimGroundingDiagnostic]:
    """Recompute claim support without changing or rejecting the supplied answer.

    This is the advisory counterpart to :func:`enforce_grounding`: callers can
    surface low confidence after delivery while the strict gate remains available
    for tests and explicit validation workflows.
    """

    recomputed = ground_answer(_grounding_text(answer), [], citations=answer.citations)
    diagnostics: list[ClaimGroundingDiagnostic] = []
    for claim in _split_claims(_grounding_text(answer)):
        if not _is_substantive(claim):
            continue
        citation_ids = tuple(recomputed.claim_map.get(claim, ()))
        diagnostics.append(
            ClaimGroundingDiagnostic(
                claim=claim,
                citation_ids=citation_ids,
                supported=bool(citation_ids),
            )
        )
    return diagnostics


def _protected_period(text: str, index: int) -> bool:
    """Return whether ``text[index]`` belongs to a decimal, URL, or abbreviation."""

    previous = text[index - 1] if index else ""
    following = text[index + 1] if index + 1 < len(text) else ""
    if previous.isdigit() and following.isdigit():
        return True
    prefix = text[max(0, index - 8) : index + 1].lower()
    if any(prefix.endswith(abbreviation) for abbreviation in _COMMON_ABBREVIATIONS):
        return True
    token_start = index
    while token_start > 0 and not text[token_start - 1].isspace():
        token_start -= 1
    token_prefix = text[token_start:index]
    if following and not following.isspace():
        if "://" in token_prefix or token_prefix.lower().startswith("www."):
            return True
        if previous.isascii() and previous.isalnum() and following.isascii() and following.isalnum():
            return True
    return False


def _inside_url_token(text: str, index: int) -> bool:
    """Protect terminal-looking punctuation embedded inside a non-whitespace URL."""

    following = text[index + 1] if index + 1 < len(text) else ""
    if not following or following.isspace():
        return False
    token_start = index
    while token_start > 0 and not text[token_start - 1].isspace():
        token_start -= 1
    prefix = text[token_start:index].lower()
    return "://" in prefix or prefix.startswith("www.")


def _split_claims(text: str) -> list[str]:
    """Split claims while retaining punctuation and protecting non-prose spans."""

    claims: list[str] = []
    start = 0
    index = 0
    code_delimiter = ""
    while index < len(text):
        char = text[index]
        if char == "`":
            run_end = index + 1
            while run_end < len(text) and text[run_end] == "`":
                run_end += 1
            delimiter = text[index:run_end]
            if not code_delimiter:
                code_delimiter = delimiter
            elif delimiter == code_delimiter:
                code_delimiter = ""
            index = run_end
            continue
        if code_delimiter:
            index += 1
            continue
        if char in "\r\n":
            claim = text[start:index].strip()
            if claim:
                claims.append(claim)
            index += 1
            while index < len(text) and text[index] in "\r\n":
                index += 1
            start = index
            continue
        if char not in _TERMINAL_PUNCTUATION:
            index += 1
            continue
        if _inside_url_token(text, index) or (char == "." and _protected_period(text, index)):
            index += 1
            continue
        end = index + 1
        while end < len(text) and text[end] in _TERMINAL_PUNCTUATION:
            end += 1
        citation_end = end
        cursor = end
        found_citation = False
        while cursor < len(text):
            while cursor < len(text) and text[cursor] in " \t":
                cursor += 1
            match = _INLINE_CITATION.match(text, cursor)
            if not match:
                break
            found_citation = True
            citation_end = match.end()
            cursor = match.end()
        if found_citation:
            end = citation_end
        claim = text[start:end].strip()
        if claim:
            claims.append(claim)
        while end < len(text) and text[end].isspace() and text[end] not in "\r\n":
            end += 1
        start = end
        index = end
    tail = text[start:].strip()
    if tail:
        claims.append(tail)
    return claims


def ground_answer(
    answer_text: str,
    evidence: list[EvidenceItem],
    *,
    citations: list[Citation] | None = None,
    overlap_threshold: float = 0.18,
    max_citations_per_claim: int = 3,
) -> GroundedAnswer:
    """Map each substantive claim to supporting citations by lexical overlap."""

    cites = citations or build_citations(evidence)
    cite_tokens = [(c, _tokens(c.snippet)) for c in cites]
    claims = _split_claims(answer_text)

    claim_map: dict[str, list[str]] = {}
    substantive = 0
    grounded = 0
    for claim in claims:
        if not _is_substantive(claim):
            continue
        substantive += 1
        claim_tok = _tokens(claim)
        inline_ids = _inline_citation_ids(claim)
        factual_heading = bool(_MARKDOWN_HEADING.match(claim))
        if factual_heading and not inline_ids:
            # A user-visible factual heading must bind deliberately. Generic
            # lexical matching is too weak here because a shared entity name can
            # otherwise make an unrelated headline look fully grounded.
            continue
        claim_overlap_threshold = max(overlap_threshold, 0.5) if factual_heading else overlap_threshold
        scored: list[tuple[str, float]] = []
        for citation, ctoks in cite_tokens:
            if not ctoks:
                continue
            if inline_ids and citation.id.lower() not in inline_ids:
                continue
            overlap = len(claim_tok & ctoks) / max(1, len(claim_tok))
            if (
                overlap >= claim_overlap_threshold
                and not _has_semantic_conflict(claim, citation.snippet)
                and _has_sufficient_cjk_support(claim, citation.snippet)
            ):
                scored.append((citation.id, overlap))
        scored.sort(key=lambda x: x[1], reverse=True)
        ids = [cid for cid, _ in scored[:max_citations_per_claim]]
        if inline_ids and {citation_id.lower() for citation_id in ids} != inline_ids:
            ids = []
        if ids:
            grounded += 1
            claim_map[claim] = ids

    coverage = (grounded / substantive) if substantive else 0.0
    return GroundedAnswer(text=answer_text, citations=cites, claim_map=claim_map, coverage=coverage)


_CONFLICT_TYPES = {
    "temporal",
    "scope_entity",
    "authority_directness",
    "internal_public_scope",
    "peer_unresolved",
}

_CONFLICT_ANALYSIS_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "evidence_conflict_analysis",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["candidates"],
            "properties": {
                "candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "claim_id",
                            "evidence_ids",
                            "same_scope",
                            "claim_scope",
                            "conflict_type",
                            "assessments",
                            "reason",
                        ],
                        "properties": {
                            "claim_id": {"type": "string"},
                            "evidence_ids": {"type": "array", "items": {"type": "string"}},
                            "same_scope": {"type": "boolean"},
                            "claim_scope": {"type": "string"},
                            "conflict_type": {"type": "string", "enum": sorted(_CONFLICT_TYPES)},
                            "assessments": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": [
                                        "evidence_id",
                                        "authority_directness",
                                        "independent_support",
                                        "effective_at",
                                    ],
                                    "properties": {
                                        "evidence_id": {"type": "string"},
                                        "authority_directness": {"type": "integer", "minimum": 0, "maximum": 3},
                                        "independent_support": {"type": "integer", "minimum": 0, "maximum": 3},
                                        "effective_at": {"type": ["number", "null"]},
                                    },
                                },
                            },
                            "reason": {"type": "string"},
                        },
                    },
                }
            },
        },
    },
}


def _conflict_candidates(answer: GroundedAnswer) -> list[dict[str, Any]]:
    """Return deterministic candidates only; no model may create a candidate."""

    citations = {citation.id: citation for citation in answer.citations}
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for claim_index, (claim, citation_ids) in enumerate(answer.claim_map.items(), start=1):
        pair_index = 0
        for left_index, left_id in enumerate(citation_ids):
            left = citations.get(left_id)
            if left is None:
                continue
            for right_id in citation_ids[left_index + 1 :]:
                right = citations.get(right_id)
                if right is None:
                    continue
                left_family = str(left.provenance_refs[0]) if left.provenance_refs else left.document_id or left.url
                right_family = str(right.provenance_refs[0]) if right.provenance_refs else right.document_id or right.url
                if left_family and left_family == right_family:
                    continue
                if not _has_evidence_pair_conflict(left.snippet, right.snippet):
                    continue
                pair = tuple(sorted((left.id, right.id)))
                key = (claim, *pair)
                if key in seen:
                    continue
                seen.add(key)
                pair_index += 1
                conflict_type = (
                    "temporal"
                    if left.effective_at is not None
                    and right.effective_at is not None
                    and left.effective_at != right.effective_at
                    else "peer_unresolved"
                )
                candidates.append(
                    {
                        "claim_id": f"claim-{claim_index}" if pair_index == 1 else f"claim-{claim_index}-{pair_index}",
                        "claim": claim,
                        "evidence_ids": pair,
                        "citations": (citations[pair[0]], citations[pair[1]]),
                        "default_type": conflict_type,
                    }
                )
    return candidates


def analyze_evidence_pack_conflicts(
    evidence: list[EvidenceItem],
    *,
    llm: Any | None = None,
    query: str = "",
) -> tuple[list[EvidenceConflict], list[dict[str, Any]]]:
    """Analyze deterministic conflicts before the composer can omit either side."""

    citations = build_citations(evidence)
    claim_map: dict[str, list[str]] = {}
    pair_index = 0
    for left_index, left in enumerate(citations):
        for right in citations[left_index + 1 :]:
            left_family = str(left.provenance_refs[0]) if left.provenance_refs else left.document_id or left.url
            right_family = str(right.provenance_refs[0]) if right.provenance_refs else right.document_id or right.url
            if left_family and left_family == right_family:
                continue
            if not _has_evidence_pair_conflict(left.snippet, right.snippet):
                continue
            pair_index += 1
            left_statement = " ".join(left.snippet.split())[:160]
            right_statement = " ".join(right.snippet.split())[:160]
            claim_map[
                f"Evidence conflict {pair_index}: {left_statement} / {right_statement}"
            ] = [left.id, right.id]
    if not claim_map:
        return [], []
    probe = GroundedAnswer(
        text="",
        citations=citations,
        claim_map=claim_map,
        coverage=1.0,
    )
    return analyze_evidence_conflicts(probe, llm=llm, temporal_query=query or None)


def _classification_prompt(candidates: list[dict[str, Any]]) -> str:
    payload = []
    for candidate in candidates:
        evidence = []
        for citation in candidate["citations"]:
            evidence.append(
                {
                    "evidence_id": citation.id,
                    "title": citation.title[:180],
                    "snippet": citation.snippet[:800],
                    "effective_at": citation.effective_at,
                    "published_at": citation.published_at,
                    "updated_at": citation.updated_at,
                    "provenance_refs": list(citation.provenance_refs),
                }
            )
        payload.append(
            {
                "claim_id": candidate["claim_id"],
                "claim": candidate["claim"],
                "evidence_ids": list(candidate["evidence_ids"]),
                "evidence": evidence,
            }
        )
    return (
        "Classify only the supplied deterministic evidence-conflict candidates. Return strict JSON. "
        "Do not add, remove, or replace evidence ids. Decide whether each pair addresses the same claim scope. "
        "Score authority/directness and independent corroboration from 0 to 3. Copy each supplied effective_at "
        "exactly, including null. Origin (external/local/memory), hit count, and repost count are never quality "
        "tie-breakers. Do not choose a winner; deterministic code will apply scope, authority/directness, temporal "
        "range, provenance completeness, then independent corroboration in that order.\n"
        + json.dumps({"candidates": payload}, ensure_ascii=False)
    )


def _score(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 3:
        raise ValueError("conflict score must be an integer in 0..3")
    return value


def _validate_classification(raw: str, candidates: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    data = json.loads(raw)
    if not isinstance(data, dict) or not isinstance(data.get("candidates"), list):
        raise ValueError("classification must contain candidates")
    expected = {candidate["claim_id"]: candidate for candidate in candidates}
    classified: dict[str, dict[str, Any]] = {}
    for item in data["candidates"]:
        if not isinstance(item, dict):
            raise ValueError("candidate classification must be an object")
        claim_id = str(item.get("claim_id") or "")
        candidate = expected.get(claim_id)
        if candidate is None or claim_id in classified:
            raise ValueError("classification contains an unknown or duplicate claim id")
        evidence_ids = item.get("evidence_ids")
        if not isinstance(evidence_ids, list) or set(map(str, evidence_ids)) != set(candidate["evidence_ids"]):
            raise ValueError("classification contains invalid evidence ids")
        if not isinstance(item.get("same_scope"), bool):
            raise ValueError("same_scope must be boolean")
        claim_scope = str(item.get("claim_scope") or "").strip()
        conflict_type = str(item.get("conflict_type") or "")
        reason = str(item.get("reason") or "").strip()
        if not claim_scope or len(claim_scope) > 240 or conflict_type not in _CONFLICT_TYPES or not reason:
            raise ValueError("classification contains invalid scope, type, or reason")
        assessments = item.get("assessments")
        if not isinstance(assessments, list) or len(assessments) != len(candidate["citations"]):
            raise ValueError("classification assessments are incomplete")
        by_id: dict[str, dict[str, Any]] = {}
        citations_by_id = {citation.id: citation for citation in candidate["citations"]}
        for assessment in assessments:
            if not isinstance(assessment, dict):
                raise ValueError("assessment must be an object")
            evidence_id = str(assessment.get("evidence_id") or "")
            citation = citations_by_id.get(evidence_id)
            if citation is None or evidence_id in by_id:
                raise ValueError("assessment contains an unknown or duplicate evidence id")
            effective_at = assessment.get("effective_at")
            if effective_at is not None and not isinstance(effective_at, (int, float)):
                raise ValueError("effective_at must be numeric or null")
            expected_at = citation.effective_at
            if (effective_at is None) != (expected_at is None) or (
                effective_at is not None and abs(float(effective_at) - float(expected_at)) > 0.001
            ):
                raise ValueError("classification changed an evidence date")
            by_id[evidence_id] = {
                "authority_directness": _score(assessment.get("authority_directness")),
                "independent_support": _score(assessment.get("independent_support")),
            }
        classified[claim_id] = {
            "same_scope": item["same_scope"],
            "claim_scope": claim_scope,
            "conflict_type": conflict_type,
            "reason": reason,
            "assessments": by_id,
        }
    if set(classified) != set(expected):
        raise ValueError("classification omitted deterministic candidates")
    return classified


def _unique_score_winner(assessments: dict[str, dict[str, int]], field: str) -> str | None:
    ranked = sorted(((values[field], evidence_id) for evidence_id, values in assessments.items()), reverse=True)
    if len(ranked) < 2 or ranked[0][0] == ranked[1][0]:
        return None
    return ranked[0][1]


def _temporal_winner(claim: str, citations: tuple[Citation, Citation]) -> str | None:
    intent, requested = detect_temporal_intent(claim)
    dated = [(citation.id, citation.effective_at) for citation in citations if citation.effective_at is not None]
    if len(dated) != len(citations):
        return None
    if intent == "latest":
        ranked = sorted(dated, key=lambda item: float(item[1]), reverse=True)
        return ranked[0][0] if ranked[0][1] != ranked[1][1] else None
    if intent == "as_of" and requested is not None:
        eligible = [item for item in dated if float(item[1]) <= requested]
        if not eligible:
            return None
        ranked = sorted(eligible, key=lambda item: float(item[1]), reverse=True)
        return ranked[0][0] if len(ranked) == 1 or ranked[0][1] != ranked[1][1] else None
    return None


def _provenance_winner(citations: tuple[Citation, Citation]) -> str | None:
    completeness = {
        citation.id: bool(
            citation.provenance_refs
            and (citation.artifact_id if citation.origin in {"local", "memory"} else citation.document_id or citation.url)
        )
        for citation in citations
    }
    complete = [evidence_id for evidence_id, value in completeness.items() if value]
    return complete[0] if len(complete) == 1 else None


def _fallback_conflicts(candidates: list[dict[str, Any]]) -> tuple[list[EvidenceConflict], list[dict[str, Any]]]:
    conflicts: list[EvidenceConflict] = []
    audit: list[dict[str, Any]] = []
    for candidate in candidates:
        conflict = EvidenceConflict(
            id=f"conflict-{len(conflicts) + 1}",
            claim_id=candidate["claim_id"],
            claim=candidate["claim"],
            type=candidate["default_type"],
            claim_scope=candidate["claim"],
            evidence_ids=candidate["evidence_ids"],
            reason="classification_unavailable; deterministic candidate remains unresolved",
        )
        conflicts.append(conflict)
        audit.append(
            {
                "claim_id": conflict.claim_id,
                "evidence_ids": list(conflict.evidence_ids),
                "resolution": "contested",
                "basis": "classification_unavailable",
                "reason": conflict.reason,
            }
        )
    return conflicts, audit


def analyze_evidence_conflicts(
    answer: GroundedAnswer,
    *,
    llm: Any | None = None,
    temporal_query: str | None = None,
) -> tuple[list[EvidenceConflict], list[dict[str, Any]]]:
    """Classify deterministic candidates once (one retry), then apply locked code rules."""

    candidates = _conflict_candidates(answer)
    if not candidates:
        answer.conflicts = []
        return [], []
    if llm is None:
        conflicts, audit = _fallback_conflicts(candidates)
    else:
        prompt = _classification_prompt(candidates)
        classified = None
        for attempt in range(2):
            try:
                result = llm.complete(
                    [{"role": "user", "content": prompt if attempt == 0 else f"Previous output was invalid.\n{prompt}"}],
                    temperature=0.1,
                    max_tokens=1200,
                    thinking_budget=0,
                    response_format=_CONFLICT_ANALYSIS_SCHEMA,
                )
                classified = _validate_classification(result.text, candidates)
                break
            except Exception:  # noqa: BLE001 - bounded fallback is part of the public contract
                continue
        if classified is None:
            conflicts, audit = _fallback_conflicts(candidates)
        else:
            conflicts = []
            audit = []
            for candidate in candidates:
                item = classified[candidate["claim_id"]]
                winner = None
                basis = ""
                if not item["same_scope"]:
                    basis = "scope_match"
                else:
                    winner = _unique_score_winner(item["assessments"], "authority_directness")
                    if winner:
                        basis = "authority_directness"
                    if not winner:
                        winner = _temporal_winner(
                            temporal_query or candidate["claim"],
                            candidate["citations"],
                        )
                        if winner:
                            basis = "temporal_scope"
                    if not winner:
                        winner = _provenance_winner(candidate["citations"])
                        if winner:
                            basis = "provenance_completeness"
                    if not winner:
                        winner = _unique_score_winner(item["assessments"], "independent_support")
                        if winner:
                            basis = "independent_corroboration"
                if basis:
                    audit.append(
                        {
                            "claim_id": candidate["claim_id"],
                            "evidence_ids": list(candidate["evidence_ids"]),
                            "resolution": "resolved",
                            "basis": basis,
                            "winner_evidence_id": winner,
                            "reason": item["reason"],
                        }
                    )
                    continue
                conflict = EvidenceConflict(
                    id=f"conflict-{len(conflicts) + 1}",
                    claim_id=candidate["claim_id"],
                    claim=candidate["claim"],
                    type=item["conflict_type"],
                    claim_scope=item["claim_scope"],
                    evidence_ids=candidate["evidence_ids"],
                    reason=item["reason"],
                )
                conflicts.append(conflict)
                audit.append(
                    {
                        "claim_id": conflict.claim_id,
                        "evidence_ids": list(conflict.evidence_ids),
                        "resolution": "contested",
                        "basis": "peer_unresolved",
                        "reason": conflict.reason,
                    }
                )
    answer.conflicts = conflicts
    if conflicts:
        answer.confidence_label = CONFIDENCE_CONTESTED
    return conflicts, audit


def detect_evidence_conflicts(answer: GroundedAnswer) -> list[EvidenceConflict]:
    """Compatibility wrapper using the conservative no-classifier fallback."""

    conflicts, _ = analyze_evidence_conflicts(answer)
    return conflicts


def enforce_grounding(answer: GroundedAnswer, *, min_coverage: float = 0.85) -> GroundedAnswer:
    """Require coverage strictly above the threshold and no unknown references."""

    unknown_citations = _unknown_citation_ids(answer)
    citation_binding_valid = _citation_bindings_valid(answer)
    zero_citable = not any(citation.grounding_eligible for citation in answer.citations)
    if zero_citable or answer.coverage <= min_coverage or unknown_citations or not citation_binding_valid:
        if zero_citable:
            reason = "zero_citable_evidence"
        elif answer.coverage <= min_coverage:
            reason = "coverage_below_threshold"
        else:
            reason = "invalid_citation_binding"
        raise BaselineError(
            ErrorCode.GROUNDING_REQUIRED,
            "Answer is insufficiently grounded in retrieved evidence.",
            details={
                "coverage": round(answer.coverage, 4),
                "min_coverage": min_coverage,
                "unknown_citations": len(unknown_citations),
                "citation_binding_valid": citation_binding_valid,
                "reason": reason,
            },
        )
    return answer
