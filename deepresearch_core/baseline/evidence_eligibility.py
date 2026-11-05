"""Run-scoped research intent and evidence eligibility for T48.

This module deliberately keeps relevance and authority decisions separate from
``EvidenceItem.grounding_eligible``.  The latter is a persistent provenance
property; the decisions below are conservative, query-scoped filters.
"""

from __future__ import annotations

import re
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import Literal, Sequence

from .evidence_trust import evidence_identity, provenance_family
from .retrieval import EvidenceItem

SourceRequirement = Literal["none", "authoritative", "primary"]
OutputShape = Literal["default", "concise", "sentence_count", "max_length"]
ParseStatus = Literal["parsed", "partial", "unchanged"]
EligibilityDecision = Literal["eligible", "rejected", "unknown"]
RelevanceStatus = Literal["strong", "weak", "unrelated", "unknown"]
AuthorityStatus = Literal["not_requested", "satisfied", "unsatisfied", "unknown"]

_WORD_NUMBER = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
    "十": 10,
}
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "answer",
    "are",
    "as",
    "at",
    "be",
    "brief",
    "by",
    "according",
    "cited",
    "concise",
    "compare",
    "compared",
    "comparison",
    "explain",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "latest",
    "not",
    "of",
    "on",
    "or",
    "our",
    "current",
    "currently",
    "now",
    "please",
    "source",
    "sources",
    "sentence",
    "sentences",
    "tell",
    "today",
    "the",
    "to",
    "use",
    "using",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "简洁",
    "简要",
    "回答",
    "什么",
    "如何",
    "请",
    "使用",
    "来源",
}
_INTERNAL_SCOPE_MARKERS = (
    "our internal",
    "our company",
    "our organization",
    "our workspace",
    "internal project",
    "内部",
    "本公司",
    "本组织",
    "我们的",
    "工作区",
)
_COUNT_PATTERN = r"\d+|one|two|three|four|five|six|seven|eight|nine|ten"
_ZH_COUNT_PATTERN = r"\d+|[一二两三四五六七八九十]"
_SOURCE_QUALITY_PATTERN = r"authoritative|official|primary|first[- ]hand"
_ZH_SOURCE_QUALITY_PATTERN = r"权威|官方|第一手|原始"
_SOURCE_OBJECT_PATTERN = r"sources?|documentation|documents?"
_ZH_SOURCE_OBJECT_PATTERN = r"来源|资料|文档"
_SOURCE_ENTITY_GAP_PATTERN = (
    r"(?:(?!(?:but|not|unverified|unofficial|non-authoritative)\b)"
    r"[A-Za-z0-9][A-Za-z0-9_.+-]*\s+){0,3}"
)
_PRIMARY_SOURCE_REQUEST_PATTERN = (
    rf"(?:primary|first[- ]hand)\s+{_SOURCE_ENTITY_GAP_PATTERN}(?:{_SOURCE_OBJECT_PATTERN})"
)
_AUTHORITATIVE_SOURCE_REQUEST_PATTERN = (
    rf"(?:authoritative|official)\s+{_SOURCE_ENTITY_GAP_PATTERN}(?:{_SOURCE_OBJECT_PATTERN})"
)
_SENTENCE_QUALIFIER_PATTERN = r"(?:(?:concise|short|cited|well[- ]cited)\s+)*"
_SOURCE_NEGATION_PATTERNS = (
    rf"\b(?:do\s+not|don't|without|no\s+need\s+to)\s+(?:use|cite|include)?\s*(?:any\s+)?(?:(?:{_COUNT_PATTERN})\s+)?(?:{_SOURCE_QUALITY_PATTERN})\s+(?:{_SOURCE_OBJECT_PATTERN})\b",
    rf"(?:请)?(?:不要|不使用|无需|不用)(?:使用|引用|采用)?\s*(?:(?:{_ZH_COUNT_PATTERN})\s*(?:个|条)?)?(?:{_ZH_SOURCE_QUALITY_PATTERN})?(?:{_ZH_SOURCE_OBJECT_PATTERN})",
)
_STANDARD_AUTHORITY_HOSTS = (
    "w3.org",
    "ietf.org",
    "rfc-editor.org",
    "iso.org",
    "iec.ch",
    "ieee.org",
    "oasis-open.org",
)
_COMMUNITY_CONTENT = re.compile(
    r"(?:^|/)(?:blogs?|community|discuss|discussions?|discourse|forums?|forumpost|news|questions?|answers?|issues?)(?:/|$)"
    r"|\b(?:blog|community|discuss|discussion|discourse|forum|news|user post|question|answer)\b",
    re.IGNORECASE,
)
_COMMUNITY_HOST_LABELS = frozenset(
    {
        "answers",
        "blog",
        "blogs",
        "community",
        "discuss",
        "discussion",
        "discussions",
        "discourse",
        "forum",
        "forums",
        "news",
        "questions",
    }
)
_PROBLEM_SOLUTION_NON_ENTITY_ANCHORS = frozenset(
    {
        "authoritative",
        "did",
        "do",
        "does",
        "documentation",
        "document",
        "documents",
        "docs",
        "official",
        "primary",
        "problem",
        "problems",
        "solve",
        "solved",
        "solves",
        "solving",
    }
)


@dataclass(frozen=True)
class ResearchIntent:
    """Deterministically separated retrieval and presentation intent."""

    original_query: str
    research_focus: str
    source_requirement: SourceRequirement = "none"
    citation_limit: int | None = None
    output_shape: OutputShape = "default"
    sentence_count: int | None = None
    max_length: int | None = None
    parse_status: ParseStatus = "unchanged"
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "original_query": self.original_query,
            "research_focus": self.research_focus,
            "source_requirement": self.source_requirement,
            "citation_limit": self.citation_limit,
            "output_shape": self.output_shape,
            "sentence_count": self.sentence_count,
            "max_length": self.max_length,
            "parse_status": self.parse_status,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class EvidenceEligibilityOutcome:
    """A safe, query-scoped eligibility decision for one evidence item."""

    evidence_id: str
    decision: EligibilityDecision
    relevance: RelevanceStatus
    authority_status: AuthorityStatus
    reason_codes: tuple[str, ...]
    matched_anchors: tuple[str, ...]
    provenance_family: str

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "decision": self.decision,
            "relevance": self.relevance,
            "authority_status": self.authority_status,
            "reason_codes": list(self.reason_codes),
            "matched_anchors": list(self.matched_anchors),
            "provenance_family": self.provenance_family,
        }


@dataclass(frozen=True)
class EvidenceAssessment:
    """Eligibility outcomes plus the candidate pack safe to adopt."""

    outcomes: tuple[EvidenceEligibilityOutcome, ...]
    adopted: tuple[EvidenceItem, ...]
    trigger: str | None = None

    @property
    def rejected_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.outcomes if item.decision == "rejected")

    @property
    def rejected_provenance_roots(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                item.provenance_family for item in self.outcomes if item.decision == "rejected"
            )
        )


@dataclass(frozen=True)
class RemediationOutcome:
    """Secret-free audit of the run's single T48 remediation allowance."""

    attempted: bool = False
    attempt_index: int = 0
    trigger: str | None = None
    action: Literal["retrieve_and_rewrite", "rewrite_only"] | None = None
    focused_query: str = ""
    excluded_evidence_ids: tuple[str, ...] = ()
    excluded_provenance_roots: tuple[str, ...] = ()
    lane_ids: tuple[str, ...] = ()
    new_evidence_ids: tuple[str, ...] = ()
    status: Literal["not_attempted", "completed", "exhausted", "failed", "timeout", "cancelled"] = (
        "not_attempted"
    )
    started_at: float | None = None
    finished_at: float | None = None
    source_generation_id: str | None = None
    replacement_generation_id: str | None = None

    def to_dict(self) -> dict:
        return {
            "attempted": self.attempted,
            "attempt_index": self.attempt_index,
            "trigger": self.trigger,
            "action": self.action,
            "focused_query": self.focused_query,
            "excluded_evidence_ids": list(self.excluded_evidence_ids),
            "excluded_provenance_roots": list(self.excluded_provenance_roots),
            "lane_ids": list(self.lane_ids),
            "new_evidence_ids": list(self.new_evidence_ids),
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "source_generation_id": self.source_generation_id,
            "replacement_generation_id": self.replacement_generation_id,
        }


def parse_research_intent(query: str) -> ResearchIntent:
    """Conservatively remove only explicit, allowlisted presentation clauses."""

    original = str(query or "").strip()
    masked = _mask_quoted(original)
    reasons: list[str] = []
    source_negated = any(
        re.search(pattern, masked, re.IGNORECASE) for pattern in _SOURCE_NEGATION_PATTERNS
    )
    source_scannable = _mask_patterns(masked, _SOURCE_NEGATION_PATTERNS)

    source_requirement: SourceRequirement = "none"
    if source_negated:
        reasons.append("source_requirement_negated")
    elif re.search(
        rf"\b(?:{_PRIMARY_SOURCE_REQUEST_PATTERN})\b",
        source_scannable,
        re.IGNORECASE,
    ) or re.search(
        rf"(?:第一手|原始)(?:{_ZH_SOURCE_OBJECT_PATTERN})", source_scannable
    ):
        source_requirement = "primary"
        reasons.append("source_requirement_primary")
    elif re.search(
        rf"\b(?:{_AUTHORITATIVE_SOURCE_REQUEST_PATTERN})\b",
        source_scannable,
        re.IGNORECASE,
    ) or re.search(
        rf"(?:权威|官方)(?:{_ZH_SOURCE_OBJECT_PATTERN})", source_scannable
    ):
        source_requirement = "authoritative"
        reasons.append("source_requirement_authoritative")

    citation_limit = _extract_bounded_count(
        source_scannable,
        (
            rf"\b(?:use|using|cite|with)\s+(?P<count>{_COUNT_PATTERN})\s+(?:(?:{_SOURCE_QUALITY_PATTERN})\s+)?(?:{_SOURCE_OBJECT_PATTERN})\b",
            rf"(?:使用|引用|采用|用|以)\s*(?P<count>{_ZH_COUNT_PATTERN})\s*(?:个|条)?(?:{_ZH_SOURCE_QUALITY_PATTERN})?(?:{_ZH_SOURCE_OBJECT_PATTERN})",
        ),
        maximum=20,
    )
    if citation_limit is not None:
        reasons.append("citation_limit")

    sentence_count = _extract_bounded_count(
        masked,
        (
            rf"\b(?:(?:answer|respond|write|return|use)\s+)?(?:(?:in|using)\s+)?"
            rf"(?:exactly\s+)?(?P<count>{_COUNT_PATTERN})\s+"
            rf"{_SENTENCE_QUALIFIER_PATTERN}sentences?\b(?:\s+with\s+citations?)?",
            rf"(?:用|以|和|及|并用|并以)?\s*(?P<count>{_ZH_COUNT_PATTERN})\s*句(?:简洁|简要|精炼)?(?:的)?(?:话|回答|答复)?",
        ),
        maximum=12,
    )
    max_length = _extract_bounded_count(
        masked,
        (
            r"\b(?:within|under|no more than)\s+(?P<count>\d+)\s+characters?\b",
            r"(?P<count>\d+)\s*(?:字|字符)(?:以内|之内)",
        ),
        maximum=20_000,
    )
    concise = bool(re.search(r"\b(?:concise|brief)\b|(?:简洁|简要)", masked, re.IGNORECASE))

    if sentence_count is not None:
        output_shape: OutputShape = "sentence_count"
        reasons.append("output_sentence_count")
    elif max_length is not None:
        output_shape = "max_length"
        reasons.append("output_max_length")
    elif concise:
        output_shape = "concise"
        reasons.append("output_concise")
    else:
        output_shape = "default"

    focus = _focus_without_constraint_sentences(original)
    parsed_any = bool(reasons)
    if not focus:
        focus = original
        status: ParseStatus = "partial" if parsed_any else "unchanged"
        reasons.append("focus_fallback_original")
    elif focus != original:
        status = "parsed"
        reasons.append("focus_constraint_removed")
    elif parsed_any:
        status = "partial"
    else:
        status = "unchanged"

    return ResearchIntent(
        original_query=original,
        research_focus=focus,
        source_requirement=source_requirement,
        citation_limit=citation_limit,
        output_shape=output_shape,
        sentence_count=sentence_count,
        max_length=max_length,
        parse_status=status,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def assess_evidence(intent: ResearchIntent, evidence: Sequence[EvidenceItem]) -> EvidenceAssessment:
    """Return conservative query-scoped eligibility without mutating evidence."""

    anchors, specific = _focus_anchors(intent.research_focus)
    definition_signal_required = _definition_signal_required(intent.research_focus)
    measurement_subjects = _measurement_subject_anchors(intent.research_focus)
    problem_solution_signal_required = _problem_solution_signal_required(intent.research_focus)
    recommendation_signal_required = _recommendation_signal_required(intent.research_focus)
    outcomes: list[EvidenceEligibilityOutcome] = []
    eligible: list[EvidenceItem] = []

    for item in evidence:
        relevance, matched = _relevance(item, anchors=anchors, specific=specific)
        definition_subject_missing = bool(
            definition_signal_required
            and any(char.isdigit() for anchor in specific for char in anchor)
            and any(
                any(char.isdigit() for char in anchor)
                and not _anchor_in_plain_text(anchor, item.text.lower())
                for anchor in specific
            )
        )
        measurement_signal_matched = bool(
            measurement_subjects
            and _supports_measurement_answer(item.text, measurement_subjects)
        )
        definition_signal_matched = bool(
            definition_signal_required
            and not definition_subject_missing
            and (_supports_definition(item.text) or measurement_signal_matched)
        )
        definition_signal_missing = bool(
            definition_signal_required
            and item.grounding_eligible
            and item.text.strip()
            and not definition_signal_matched
        )
        if (
            definition_signal_matched
            and relevance in {"weak", "unknown"}
            and (specific.intersection(matched) or measurement_signal_matched)
        ):
            relevance = "strong"
        if definition_signal_missing and relevance == "strong":
            relevance = "weak"
        problem_solution_signal_matched = bool(
            problem_solution_signal_required
            and _supports_problem_solution_for_item(
                item,
                anchors=anchors,
                matched=matched,
            )
        )
        if problem_solution_signal_required:
            if problem_solution_signal_matched and relevance in {"weak", "unknown"}:
                relevance = "strong"
            elif not problem_solution_signal_matched and relevance == "strong":
                relevance = "weak"
        recommendation_signal_matched = bool(
            recommendation_signal_required
            and _supports_recommendation_for_item(
                item,
                anchors=anchors,
                focus=intent.research_focus,
            )
        )
        if recommendation_signal_required:
            if recommendation_signal_matched and relevance in {"weak", "unknown"}:
                relevance = "strong"
            elif not recommendation_signal_matched and relevance == "strong":
                relevance = "weak"
        authority = _authority_status(intent, item, relevance=relevance, anchors=anchors, matched=matched)
        reasons: list[str] = []
        if definition_signal_missing:
            reasons.append("definition_signal_missing")
        if definition_subject_missing:
            reasons.append("definition_subject_missing")
        if problem_solution_signal_required:
            reasons.append(
                "problem_solution_signal_matched"
                if problem_solution_signal_matched
                else "problem_solution_signal_missing"
            )
        if recommendation_signal_required:
            reasons.append(
                "recommendation_signal_matched"
                if recommendation_signal_matched
                else "recommendation_signal_missing"
            )
        if not item.grounding_eligible or not item.text.strip():
            decision: EligibilityDecision = "rejected"
            reasons.append("not_citable")
        elif relevance == "unrelated":
            decision = "rejected"
            reasons.append("research_focus_unrelated")
        elif authority == "unsatisfied":
            decision = "rejected"
            reasons.append("authority_requirement_unsatisfied")
        elif relevance == "strong" and authority in {"not_requested", "satisfied"}:
            decision = "eligible"
            reasons.append("strong_focus_match")
            if authority == "satisfied":
                reasons.append("authority_requirement_satisfied")
        else:
            decision = "unknown"
            reasons.append("insufficient_high_confidence_signal")
        outcome = EvidenceEligibilityOutcome(
            evidence_id=evidence_identity(item),
            decision=decision,
            relevance=relevance,
            authority_status=authority,
            reason_codes=tuple(reasons),
            matched_anchors=tuple(matched[:12]),
            provenance_family=provenance_family(item),
        )
        outcomes.append(outcome)
        if decision == "eligible":
            eligible.append(item)

    if eligible:
        adopted = list(eligible)
        trigger = None
    else:
        adopted = []
        if not evidence or not any(item.grounding_eligible and item.text.strip() for item in evidence):
            trigger = "zero_citable_evidence"
        elif intent.source_requirement != "none" and any(
            item.relevance in {"strong", "weak"} for item in outcomes
        ):
            trigger = "authority_unmet"
        elif outcomes and all(item.decision == "rejected" for item in outcomes):
            trigger = "prewriter_irrelevant"
        else:
            trigger = "unknown_only"
    return EvidenceAssessment(outcomes=tuple(outcomes), adopted=tuple(adopted), trigger=trigger)


def apply_citation_limit(
    intent: ResearchIntent,
    evidence: Sequence[EvidenceItem],
    *,
    conflicts_present: bool,
) -> list[EvidenceItem]:
    """Apply a public citation limit only after conflict analysis."""

    items = list(evidence)
    if conflicts_present or intent.citation_limit is None:
        return items
    return items[: max(1, intent.citation_limit)]


def output_instruction(intent: ResearchIntent) -> str:
    """Render only typed presentation constraints for the untrusted user prompt."""

    parts: list[str] = []
    shape_constrained = intent.output_shape in {"sentence_count", "max_length", "concise"}
    if intent.output_shape == "sentence_count" and intent.sentence_count:
        parts.append(f"Return exactly {intent.sentence_count} concise sentences of answer text.")
    elif intent.output_shape == "max_length" and intent.max_length:
        parts.append(f"Keep the answer text within {intent.max_length} characters.")
    elif intent.output_shape == "concise":
        parts.append("Keep the answer concise.")
    if intent.citation_limit:
        parts.append(f"Use no more than {intent.citation_limit} public citation source(s).")
    if shape_constrained:
        parts.append("Place inline [cN] citations inside those sentences; do not add headings or extra sections.")
    elif intent.citation_limit:
        parts.append("Place inline [cN] citations next to the claims they support.")
    return " ".join(parts)


def output_shape_valid(intent: ResearchIntent, text: str) -> bool:
    """Validate deterministic presentation constraints before final persistence."""

    if intent.output_shape == "sentence_count" and intent.sentence_count:
        return _sentence_count(text) == intent.sentence_count
    if intent.output_shape == "max_length" and intent.max_length:
        visible = re.sub(r"\[c\d+\]", "", text).strip()
        return len(visible) <= intent.max_length
    return True


def focused_recovery_query(intent: ResearchIntent, trigger: str | None) -> str:
    """Return a bounded, typed recovery query without leaking audit details."""

    suffix = ""
    if intent.source_requirement == "primary":
        suffix = " primary source direct documentation"
    elif intent.source_requirement == "authoritative":
        suffix = " authoritative official documentation"
    elif trigger in {"prewriter_irrelevant", "unknown_only", "zero_citable_evidence"}:
        suffix = " direct documentation"
    standards_hint = (
        " IETF RFC"
        if re.search(r"\bHTTP\s+(?:status\s+)?\d{3}\b", intent.research_focus, re.IGNORECASE)
        else ""
    )
    return f"{intent.research_focus}{standards_hint}{suffix}".strip()


def started_remediation(
    *,
    trigger: str,
    action: Literal["retrieve_and_rewrite", "rewrite_only"],
    focused_query: str,
    excluded_evidence_ids: Sequence[str] = (),
    excluded_provenance_roots: Sequence[str] = (),
    source_generation_id: str | None = None,
) -> RemediationOutcome:
    """Create the one allowed in-progress remediation audit record."""

    return RemediationOutcome(
        attempted=True,
        attempt_index=1,
        trigger=trigger,
        action=action,
        focused_query=focused_query,
        excluded_evidence_ids=tuple(excluded_evidence_ids),
        excluded_provenance_roots=tuple(excluded_provenance_roots),
        status="not_attempted",
        started_at=time.time(),
        source_generation_id=source_generation_id,
    )


def _extract_bounded_count(text: str, patterns: Sequence[str], *, maximum: int) -> int | None:
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        raw = match.group("count").lower()
        try:
            value = int(raw)
        except ValueError:
            value = _WORD_NUMBER.get(raw, 0)
        if 1 <= value <= maximum:
            return value
    return None


def _mask_quoted(text: str) -> str:
    return re.sub(r'("[^"\n]*"|“[^”\n]*”|‘[^’\n]*’|\'[^\'\n]*\')', lambda m: " " * len(m.group(0)), text)


def _mask_patterns(text: str, patterns: Sequence[str]) -> str:
    masked = text
    for pattern in patterns:
        masked = re.sub(pattern, lambda match: " " * len(match.group(0)), masked, flags=re.IGNORECASE)
    return masked


def _focus_without_constraint_sentences(query: str) -> str:
    quoted = re.compile(r'("[^"\n]*"|“[^”\n]*”|‘[^’\n]*’|\'[^\'\n]*\')')

    def clean_unquoted(text: str) -> str:
        patterns = (
            *_SOURCE_NEGATION_PATTERNS,
            rf"\b(?:please\s+)?(?:use|using|cite|with)\s+(?:(?:{_COUNT_PATTERN})\s+)?"
            rf"(?:(?:{_SOURCE_QUALITY_PATTERN})\s+{_SOURCE_ENTITY_GAP_PATTERN})?"
            rf"(?:{_SOURCE_OBJECT_PATTERN})\b",
            rf"(?:请)?(?:使用|引用|采用|用|以)\s*(?:(?:{_ZH_COUNT_PATTERN})\s*(?:个|条)?)?(?:{_ZH_SOURCE_QUALITY_PATTERN})?(?:{_ZH_SOURCE_OBJECT_PATTERN})",
            rf"\b(?:(?:answer|respond|write|return|use)\s+)?(?:(?:in|using)\s+)?"
            rf"(?:exactly\s+)?(?:{_COUNT_PATTERN})\s+{_SENTENCE_QUALIFIER_PATTERN}"
            rf"sentences?\b(?:\s+with\s+citations?)?",
            rf"(?:并|和|及)?(?:用|以)?\s*(?:{_ZH_COUNT_PATTERN})\s*句(?:简洁|简要|精炼)?(?:的)?(?:话|回答|答复)?",
            r"\b(?:please\s+)?(?:answer|respond|write)\s+(?:concisely|briefly)\b",
            r"(?:请)?(?:简洁|简要)(?:回答|答复)",
            r"\b(?:within|under|no more than)\s+\d+\s+characters?\b",
            r"\d+\s*(?:字|字符)(?:以内|之内)",
        )
        cleaned = text
        for pattern in patterns:
            cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+(?=[,.;!?，。；！？])", "", cleaned)
        cleaned = re.sub(r"(?:^|(?<=[.!?。！？]))\s*(?:please|and|并|和|及)[,，]?\s*(?=$|[.!?。！？])", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s{2,}", " ", cleaned)
        return cleaned

    parts = quoted.split(query)
    cleaned = "".join(part if quoted.fullmatch(part or "") else clean_unquoted(part) for part in parts)
    pieces = [
        match.group(0).strip()
        for match in re.finditer(
            r".+?(?:[!?。！？]|(?:(?<![A-Za-z0-9_])\.|\.(?![A-Za-z0-9_]))|$)",
            cleaned,
        )
        if re.search(r"[A-Za-z0-9\u3400-\u9fff]", match.group(0))
    ]
    return " ".join(pieces).strip()


def _focus_anchors(focus: str) -> tuple[list[str], set[str]]:
    normalized_focus = re.sub(r"\b\d{4}-\d{1,2}-\d{1,2}\b", " ", focus)
    normalized_focus = re.sub(
        r"\b(?:as\s+of|through|up\s+to|until)\s+(?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{4}|\d{4}(?:-\d{1,2})?)\b",
        " ",
        normalized_focus,
        flags=re.IGNORECASE,
    )
    normalized_focus = re.sub(
        r"(?:截至|截止|到)\s*\d{4}年(?:\d{1,2}月(?:\d{1,2}日)?)?",
        " ",
        normalized_focus,
    )
    negated = {
        token.lower().strip("._+-")
        for token in re.findall(
            r"\b(?:not|excluding|except)\s+([A-Za-z][A-Za-z0-9_+-]*)",
            normalized_focus,
            re.IGNORECASE,
        )
    }
    raw_tokens = re.findall(
        r"v?\d+(?:\.\d+)+|[A-Za-z][A-Za-z0-9_+-]*|\d+|[\u3400-\u9fff]{2,}",
        normalized_focus,
        re.IGNORECASE,
    )
    anchors: list[str] = []
    specific: set[str] = set()
    for raw in raw_tokens:
        if re.fullmatch(r"[\u3400-\u9fff]{2,}", raw):
            for part in _cjk_focus_parts(raw):
                if part not in anchors:
                    anchors.append(part)
            continue
        lowered = raw.lower().strip("._+-")
        if not lowered or lowered in _STOP_WORDS or lowered in negated or len(lowered) < 2:
            continue
        if lowered not in anchors:
            anchors.append(lowered)
        if (
            any(char.isdigit() for char in lowered)
            or (raw.isupper() and len(raw) >= 2)
            or sum(1 for char in raw[1:] if char.isupper()) >= 2
        ):
            specific.add(lowered)
    return anchors[:16], specific


def _definition_signal_required(focus: str) -> bool:
    return bool(
        re.search(
            r"\b(?:definition|define)\b|\bwhat\s+does\b.{1,100}\b(?:mean|measure)\b|"
            r"定义|含义|(?:衡量|测量)什么",
            str(focus or ""),
            re.IGNORECASE,
        )
    )


def _supports_definition(text: str) -> bool:
    return bool(
        re.search(
            r"\bdefin(?:e|es|ed|ition)\b|\bmeans?\b|\brefers?\s+to\b|"
            r"\b(?:indicates?|represents?|instantiates?)\b|定义为|是指|含义是",
            str(text or ""),
            re.IGNORECASE,
        )
    )


def _measurement_subject_anchors(focus: str) -> tuple[str, ...]:
    normalized = str(focus or "")
    match = re.search(
        r"\bwhat\s+does\s+(.{1,100}?)\s+measure\b",
        normalized,
        re.IGNORECASE,
    )
    if match:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_+-]*|\d+", match.group(1))
        return tuple(
            dict.fromkeys(
                token.lower().strip("._+-")
                for token in tokens
                if token.lower().strip("._+-") not in _STOP_WORDS
            )
        )
    cjk_match = re.search(r"([^，。？！]{1,40}?)(?:衡量|测量)什么", normalized)
    if cjk_match:
        return tuple(_cjk_focus_parts(cjk_match.group(1)))
    return ()


def _supports_measurement_answer(text: str, subjects: Sequence[str]) -> bool:
    for sentence in re.split(r"(?<=[.!?。！？])(?:\s+|$)|[\r\n]+", str(text or "")):
        lowered = sentence.lower()
        for subject in subjects:
            match = re.search(
                rf"(?<![a-z0-9_]){re.escape(subject)}(?![a-z0-9_])",
                lowered,
            )
            if not match:
                continue
            tail = lowered[match.end() : match.end() + 120]
            answer = re.search(
                r"\b(?:measures?|refers?\s+to|represents?|captures?|indicates?)\b|"
                r"(?:衡量|测量|是指|表示|反映)",
                tail,
                re.IGNORECASE,
            )
            if not answer:
                continue
            modifier = tail[: answer.start()]
            if re.search(
                r"\b(?:not|never|cannot|can't|doesn't|don't|isn't|aren't)\b|"
                r"\b(?:does|do|did|is|are|was|were)\s+not\b|"
                r"(?:不|不能|无法|并非|不是)",
                modifier,
                re.IGNORECASE,
            ):
                continue
            return True
    return False


def _recommendation_signal_required(focus: str) -> bool:
    normalized = str(focus or "")
    return bool(
        re.search(
            r"\bhow\s+(?:much|many)\b.{0,120}\b(?:should|need|get)\b|"
            r"\b(?:guidance|recommendations?)\b.{0,120}\bhow\s+(?:much|many)\b|"
            r"(?:应该|应当|建议).{0,60}(?:多少|多久)|(?:多少|多久).{0,60}(?:应该|应当|建议)",
            normalized,
            re.IGNORECASE,
        )
    )


def _supports_recommendation_for_item(
    item: EvidenceItem,
    *,
    anchors: Sequence[str],
    focus: str,
) -> bool:
    generic = {
        "according",
        "guidance",
        "recommendation",
        "recommendations",
        "much",
        "many",
        "should",
        "need",
        "get",
    }
    explicit_subjects = _recommendation_subject_anchors(focus)
    subject_anchors = list(explicit_subjects) or [
        anchor for anchor in anchors if anchor not in generic
    ]
    for sentence in re.split(r"(?<=[.!?。！？])(?:\s+|$)|[\r\n]+", str(item.text or "")):
        lowered = sentence.lower()
        has_amount = bool(
            re.search(
                r"\b(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b|"
                r"[一二三四五六七八九十两\d]+",
                lowered,
                re.IGNORECASE,
            )
        )
        has_unit = bool(
            re.search(
                r"\b(?:hours?|minutes?|days?|weeks?|servings?|grams?|milligrams?|liters?|litres?)\b|"
                r"(?:小时|分钟|天|周|份|克|毫克|升)",
                lowered,
                re.IGNORECASE,
            )
        )
        has_guidance = bool(
            re.search(
                r"\b(?:should|needs?|recommended?|at\s+least|or\s+more|per\s+(?:night|day|week))\b|"
                r"(?:应该|应当|建议|至少|不少于|每晚|每天|每周)",
                lowered,
                re.IGNORECASE,
            )
        )
        if not (has_amount and has_unit and has_guidance):
            continue
        if any(
            _anchor_in_plain_text(anchor, lowered)
            or (
                anchor.endswith("s")
                and len(anchor) > 3
                and _anchor_in_plain_text(anchor[:-1], lowered)
            )
            for anchor in subject_anchors
        ):
            return True
    return False


def _recommendation_subject_anchors(focus: str) -> tuple[str, ...]:
    """Extract the recipient entity from common amount-guidance questions."""

    normalized = str(focus or "")
    patterns = (
        r"\bshould\s+(?P<subject>[^?.,;]{1,60}?)\s+"
        r"(?:get|have|consume|drink|eat|take|receive)\b",
        r"\b(?:do|does|did)\s+(?P<subject>[^?.,;]{1,60}?)\s+need\b",
    )
    for pattern in patterns:
        match = re.search(pattern, normalized, re.IGNORECASE)
        if not match:
            continue
        tokens = [
            token.lower().strip("._+-")
            for token in re.findall(
                r"[A-Za-z][A-Za-z0-9_+-]*|\d+|[\u3400-\u9fff]{2,}",
                match.group("subject"),
            )
        ]
        subjects = [
            token
            for token in tokens
            if token
            and token not in _STOP_WORDS
            and token not in {"most", "typically", "generally"}
        ]
        if subjects:
            return tuple(dict.fromkeys(subjects))
    return ()


def _problem_solution_signal_required(focus: str) -> bool:
    return bool(
        re.search(
            r"\bwhat\s+problem\s+does\b.{1,100}\bsolve\b|\bproblem\b.{1,100}\bsolves?\b|解决(?:了)?什么问题|解决.+问题",
            str(focus or ""),
            re.IGNORECASE,
        )
    )


def _supports_problem_solution(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:solves?|addresses?|offers?|provides?|enables?|allows?|helps?|represents?)\b|"
            r"用于|提供|支持|允许|帮助|解决",
            str(text or ""),
            re.IGNORECASE,
        )
    )


def _supports_problem_solution_for_item(
    item: EvidenceItem,
    *,
    anchors: Sequence[str],
    matched: Sequence[str],
) -> bool:
    """Require a capability statement to bind to the query's actual subject."""

    entity_anchors = [
        anchor
        for anchor in anchors
        if anchor not in _PROBLEM_SOLUTION_NON_ENTITY_ANCHORS
    ]
    if not entity_anchors or not set(entity_anchors).intersection(matched):
        return False
    if not _supports_problem_solution(item.text):
        return False
    identity = " ".join((str(item.title or ""), str(item.url or ""))).lower()
    if any(_anchor_in_plain_text(anchor, identity) for anchor in entity_anchors):
        return True
    text = str(item.text or "")
    if any(_anchor_in_plain_text(anchor, text.lower()) for anchor in entity_anchors) and re.search(
        r"\b(?:this|the)\s+(?:module|library|package|tool|system|service|framework|api|product)\b"
        r"(?!['’])"
        r".{0,120}\b(?:solves?|addresses|offers?|provides?|enables?|allows?|helps?)\b"
        r"|(?:该|这个|此)(?:模块|库|软件包|工具|系统|服务|框架|接口|产品).{0,80}"
        r"(?:解决|应对|提供|支持|允许|帮助)",
        text,
        re.IGNORECASE,
    ):
        return True
    for sentence in re.split(r"(?<=[.!?。！？])(?:\s+|$)|[\r\n]+", text):
        if _supports_problem_solution(sentence) and any(
            _anchor_in_plain_text(anchor, sentence.lower()) for anchor in entity_anchors
        ):
            return True
    return False


def _anchor_in_plain_text(anchor: str, text: str) -> bool:
    if re.fullmatch(r"[\u3400-\u9fff]+", anchor):
        return anchor in text
    return bool(re.search(rf"(?<![a-z0-9_]){re.escape(anchor)}(?![a-z0-9_])", text))


def _relevance(
    item: EvidenceItem,
    *,
    anchors: Sequence[str],
    specific: set[str],
) -> tuple[RelevanceStatus, list[str]]:
    if not anchors:
        return "unknown", []
    blob = " ".join((str(item.title or ""), str(item.url or ""), str(item.text or ""))).lower()
    tokens = set(
        re.findall(r"v?\d+(?:\.\d+)+|[a-z][a-z0-9_+-]*|\d+|[\u3400-\u9fff]{2,}", blob)
    )
    matched = [
        anchor
        for anchor in anchors
        if _anchor_matches(anchor, tokens, blob=blob) or _alias_match(anchor, blob)
    ]
    matched_specific = specific.intersection(matched)
    if len(matched) == len(anchors) or (
        specific
        and matched_specific
        and len(matched) >= min(2, len(anchors))
    ) or (
        not specific and len(matched) >= max(1, (len(anchors) + 1) // 2)
    ):
        return "strong", matched
    if matched:
        if specific and matched_specific != specific and len(item.text.strip()) >= 20:
            return "unrelated", matched
        return "weak", matched
    # A zero lexical match can be a genuine synonym.  Only a conflicting
    # entity/version signal after a partial match is strong enough to reject.
    return "unknown", []


def _anchor_matches(anchor: str, tokens: set[str], *, blob: str) -> bool:
    if re.fullmatch(r"[\u3400-\u9fff]{2,}", anchor):
        return anchor in blob
    if anchor in tokens:
        return True
    if anchor.endswith("s") and anchor[:-1] in tokens:
        return True
    return f"{anchor}s" in tokens


def _cjk_focus_parts(raw: str) -> list[str]:
    """Extract conservative Chinese concept spans without a segmentation dependency."""

    text = raw
    text = re.sub(r"(?:不是|而非|不要|排除)[\u3400-\u9fff]{2,12}$", "", text)
    for filler in (
        "请问",
        "什么是",
        "是什么",
        "请解释",
        "解释",
        "请介绍",
        "介绍",
        "请说明",
        "说明",
        "为什么",
        "如何",
        "比较",
        "对比",
    ):
        text = text.replace(filler, "")
    text = re.sub(r"(?:有什么)?(?:区别|差异)$", "", text)
    return [part for part in re.split(r"(?:和|与|及|还是)", text) if len(part) >= 2]


def _alias_match(anchor: str, blob: str) -> bool:
    match = re.fullmatch(r"([a-z]{2,})(\d+)", anchor)
    if not match:
        return False
    letters, version = match.groups()
    words = re.findall(r"[a-z]+", blob)
    for index in range(0, max(0, len(words) - len(letters) + 1)):
        window = words[index : index + len(letters)]
        if "".join(word[0] for word in window) != letters:
            continue
        nearby = " ".join(words[index : index + len(letters) + 4])
        if re.search(rf"(?:version\s*)?{re.escape(version)}\b", blob) or version in nearby:
            return True
    phrase = r"\b" + r"[-\s]+".join(re.escape(letter) + r"[a-z]*" for letter in letters) + r"\b"
    return bool(re.search(phrase, blob) and re.search(rf"\b(?:version\s*)?{re.escape(version)}\b", blob))


def _authority_status(
    intent: ResearchIntent,
    item: EvidenceItem,
    *,
    relevance: RelevanceStatus,
    anchors: Sequence[str],
    matched: Sequence[str],
) -> AuthorityStatus:
    if intent.source_requirement == "none":
        return "not_requested"
    if relevance not in {"strong", "weak"}:
        return "unknown"
    query_lower = intent.research_focus.lower()
    internal_scope = any(marker in query_lower for marker in _INTERNAL_SCOPE_MARKERS)
    if item.origin in {"local", "memory"}:
        uncertain_record = bool(
            re.search(
                r"\b(?:rumou?r|hearsay|unconfirmed|informal|maybe|may|speculative)\b"
                r"|(?:传闻|听说|可能|非正式|未确认)",
                f"{item.title} {item.text}",
                re.IGNORECASE,
            )
        )
        direct_internal_record = bool(
            re.search(
                r"\b(?:decision|record|specification|policy|minutes|status|launch target|source data)\b"
                r"|(?:决策|记录|规范|政策|会议纪要|状态|目标|原始数据)",
                str(item.title or ""),
                re.IGNORECASE,
            )
        )
        return (
            "satisfied"
            if internal_scope and relevance == "strong" and direct_internal_record and not uncertain_record
            else "unknown"
        )
    host = urllib.parse.urlparse(item.url).hostname or ""
    parsed_url = urllib.parse.urlparse(item.url)
    registered_label = _registered_domain_label(host)
    entity_anchors = [
        anchor
        for anchor in anchors
        if re.fullmatch(r"[a-z][a-z0-9]+", anchor) and not any(char.isdigit() for char in anchor)
    ]
    direct_entity_host = bool(registered_label and registered_label in entity_anchors)
    public_authority = host.endswith((".gov", ".gov.cn", ".edu", ".ac.uk", ".int"))
    if relevance == "strong" and _is_direct_standard_document(
        host,
        path=parsed_url.path,
        title=str(item.title or ""),
        text=str(item.text or ""),
    ):
        return "satisfied"
    community_content = bool(
        set(host.lower().strip(".").split(".")) & _COMMUNITY_HOST_LABELS
        or _COMMUNITY_CONTENT.search(f"{parsed_url.path} {str(item.title or '')}")
    )
    if relevance == "strong" and direct_entity_host and not community_content:
        return "satisfied"
    if (
        intent.source_requirement == "authoritative"
        and relevance == "strong"
        and not community_content
        and _is_direct_reference_document(host, path=parsed_url.path, title=str(item.title or ""))
    ):
        return "satisfied"
    if intent.source_requirement == "authoritative" and relevance == "strong" and public_authority:
        return "satisfied"
    if relevance == "strong" and matched:
        return "unsatisfied"
    return "unknown"


def _registered_domain_label(host: str) -> str:
    """Return the registrable-domain label used for conservative identity checks."""

    labels = [label for label in host.lower().strip(".").split(".") if label]
    if len(labels) < 2:
        return labels[0] if labels else ""
    common_second_level_suffixes = {
        "ac.uk",
        "co.jp",
        "co.uk",
        "com.au",
        "com.br",
        "com.cn",
        "com.sg",
        "edu.cn",
        "gov.cn",
        "net.cn",
        "org.cn",
        "org.uk",
    }
    suffix = ".".join(labels[-2:])
    return labels[-3] if suffix in common_second_level_suffixes and len(labels) >= 3 else labels[-2]


def _is_direct_standard_document(host: str, *, path: str, title: str, text: str) -> bool:
    """Recognize normative documents from cross-vendor standards authorities."""

    normalized_host = host.lower().strip(".")
    if not any(
        normalized_host == authority or normalized_host.endswith(f".{authority}")
        for authority in _STANDARD_AUTHORITY_HOSTS
    ):
        return False
    identity = f"{path} {title} {text[:240]}"
    return bool(
        re.search(
            r"(?:/TR/|/rfc/rfc\d+|\bRFC\s*\d+\b|\b(?:standard|specification|recommendation)\b|\bmodule\s+level\s+\d+\b)",
            identity,
            re.IGNORECASE,
        )
    )


def _is_direct_reference_document(host: str, *, path: str, title: str) -> bool:
    """Recognize publisher-owned reference/documentation surfaces generically."""

    labels = set(host.lower().strip(".").split("."))
    normalized_title = title.lower()
    publisher_host = bool(
        labels & {"developer", "developers", "docs", "documentation", "reference"}
    )
    strong_reference_path = bool(
        re.search(
            r"/(?:documentation|reference|api-reference|manual|specification|standard)(?:/|$)",
            path,
            re.IGNORECASE,
        )
    )
    reference_surface = bool(
        publisher_host
        or re.search(r"/(?:docs?|documentation|reference|api-reference)(?:/|$)", path, re.IGNORECASE)
    )
    reference_identity = bool(
        publisher_host
        or strong_reference_path
        or re.search(
            r"\b(?:docs?|documentation|reference|manual|specification|standard)\b",
            normalized_title,
            re.IGNORECASE,
        )
    )
    return reference_surface and reference_identity


def _sentence_count(text: str) -> int:
    visible = re.sub(r"\[c\d+\]", "", text).strip()
    visible = re.sub(r"```.*?```", " code ", visible, flags=re.DOTALL)
    visible = re.sub(r"`[^`\r\n]*`", " code ", visible)
    parts = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+|(?<=[。！？])", visible) if part.strip()]
    return len(parts)
