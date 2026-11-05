"""T46 evidence retrieval, temporal coverage, and conflict value contracts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from .retrieval import EvidenceItem

EvidenceLane = Literal["local", "memory", "external"]
RetrievalStatus = Literal["completed", "empty", "failed", "timeout", "cancelled"]
TemporalIntent = Literal["none", "latest", "as_of"]
TemporalStatus = Literal["not_requested", "bounded", "partially_bounded", "unknown"]
TemporalBasis = Literal["claim_effective_at", "source_published_or_updated_at", "none"]
ConflictType = Literal[
    "temporal",
    "scope_entity",
    "authority_directness",
    "internal_public_scope",
    "peer_unresolved",
]

_AS_OF_ISO = re.compile(r"\b(?:as\s+of|at)\s+(\d{4}-\d{1,2}-\d{1,2})\b", re.IGNORECASE)
_AS_OF_CJK = re.compile(r"(?:截至|截止|于)\s*(\d{4})[年/-](\d{1,2})[月/-](\d{1,2})日?")
_LATEST_ENGLISH = re.compile(
    r"\b(?:latest|current|currently|today|now|most\s+recent)\b",
    re.IGNORECASE,
)
_LATEST_CJK_MARKERS = (
    "最新",
    "当前",
    "今天",
    "今日",
    "现状",
)


@dataclass(frozen=True)
class RetrievalOutcome:
    """Immutable outcome for one independently attempted evidence lane."""

    lane_id: str
    lane: EvidenceLane
    query: str
    attempted: bool
    status: RetrievalStatus
    provider: str | None = None
    search_provider: str | None = None
    fetch_providers: tuple[str, ...] | list[str] = field(default_factory=tuple)
    providers_attempted: tuple[str, ...] | list[str] = field(default_factory=tuple)
    providers_skipped: tuple[str, ...] | list[str] = field(default_factory=tuple)
    provider_failure_reasons: dict[str, str] = field(default_factory=dict)
    tier: str | None = None
    hit_count: int = 0
    evidence_ids: tuple[str, ...] | list[str] = field(default_factory=tuple)
    error_code: str | None = None
    reason: str | None = None
    locale: dict | None = None
    search_options: dict | None = None
    started_at: float = 0.0
    finished_at: float = 0.0

    def to_dict(self) -> dict:
        payload = {
            "lane_id": self.lane_id,
            "lane": self.lane,
            "query": self.query,
            "attempted": self.attempted,
            "status": self.status,
            "provider": self.provider,
            "search_provider": self.search_provider,
            "fetch_providers": list(self.fetch_providers) or None,
            "providers_attempted": list(self.providers_attempted) or None,
            "providers_skipped": list(self.providers_skipped) or None,
            "provider_failure_reasons": dict(self.provider_failure_reasons) or None,
            "tier": self.tier,
            "hit_count": int(self.hit_count),
            "evidence_ids": list(self.evidence_ids),
            "error_code": self.error_code,
            "reason": self.reason,
            "locale": dict(self.locale) if self.locale is not None else None,
            "search_options": (
                dict(self.search_options) if self.search_options is not None else None
            ),
            "started_at": float(self.started_at),
            "finished_at": float(self.finished_at),
        }
        return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True)
class TemporalCoverage:
    """Deterministic disclosure of the fact-time boundary supported by evidence."""

    intent: TemporalIntent
    checked_at: float
    status: TemporalStatus
    basis: TemporalBasis
    evidence_ids: tuple[str, ...] | list[str] = field(default_factory=tuple)
    reason_codes: tuple[str, ...] | list[str] = field(default_factory=tuple)
    statement: str = ""
    note_required: bool = False
    requested_as_of: float | None = None
    supported_through: float | None = None

    def to_dict(self) -> dict:
        payload = {
            "intent": self.intent,
            "requested_as_of": self.requested_as_of,
            "checked_at": float(self.checked_at),
            "status": self.status,
            "supported_through": self.supported_through,
            "basis": self.basis,
            "evidence_ids": list(self.evidence_ids),
            "reason_codes": list(self.reason_codes),
            "statement": self.statement,
            "note_required": self.note_required,
        }
        return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True)
class EvidenceConflict:
    """One real, claim-scoped conflict that remains unresolved after deterministic rules."""

    id: str
    claim_id: str
    claim: str
    type: ConflictType
    claim_scope: str
    evidence_ids: tuple[str, ...] | list[str]
    resolution: Literal["contested"] = "contested"
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "claim_id": self.claim_id,
            "claim": self.claim,
            "type": self.type,
            "claim_scope": self.claim_scope,
            "evidence_ids": list(self.evidence_ids),
            "resolution": self.resolution,
            "reason": self.reason,
        }


def evidence_identity(item: EvidenceItem) -> str:
    """Return the stable evidence identity used across lanes, citations, KG, and UI."""

    if item.origin in {"local", "memory"} and item.artifact_id:
        return f"{item.origin}:{item.artifact_type}:{item.artifact_id}"
    if item.provenance_refs:
        return str(item.provenance_refs[0])
    if item.document_id:
        return f"external:web_document:{item.document_id}"
    if item.url:
        return f"external:web_document:{item.url}"
    return f"evidence:{item.chunk_id}"


def provenance_family(item: EvidenceItem) -> str:
    """Return the canonical family key without using origin as a quality preference."""

    return str(item.provenance_refs[0]) if item.provenance_refs else evidence_identity(item)


def detect_temporal_intent(query: str) -> tuple[TemporalIntent, float | None]:
    """Detect latest/current intent or an explicit historical/future as-of date."""

    text = " ".join(str(query or "").split())
    match = _AS_OF_ISO.search(text)
    if match:
        parsed = _parse_date(match.group(1))
        if parsed is not None:
            return "as_of", parsed
    match = _AS_OF_CJK.search(text)
    if match:
        parsed = _parse_date("-".join(match.groups()))
        if parsed is not None:
            return "as_of", parsed
    if _LATEST_ENGLISH.search(text) or any(marker in text for marker in _LATEST_CJK_MARKERS):
        return "latest", None
    return "none", None


def temporal_validation_query(query: str, *, intent: TemporalIntent, requested_as_of: float | None) -> str:
    """Build the one supplemental assurance query without imposing an arbitrary freshness window."""

    if intent == "as_of" and requested_as_of is not None:
        return f"{query} status effective as of {_date_label(requested_as_of)}"
    if intent == "latest":
        return f"{query} latest current status effective date"
    return query


def assess_temporal_coverage(
    *,
    intent: TemporalIntent,
    requested_as_of: float | None,
    evidence: list[EvidenceItem],
    checked_at: float,
) -> TemporalCoverage:
    """Assess fact-time coverage; retrieval time is intentionally never a supporting basis."""

    eligible = [item for item in evidence if item.grounding_eligible]
    if intent == "none":
        return TemporalCoverage(
            intent="none",
            checked_at=checked_at,
            status="not_requested",
            basis="none",
        )

    effective = [(item, item.effective_at) for item in eligible if item.effective_at is not None]
    source_dated = [
        (item, max(value for value in (item.published_at, item.updated_at) if value is not None))
        for item in eligible
        if item.published_at is not None or item.updated_at is not None
    ]
    if effective:
        basis: TemporalBasis = "claim_effective_at"
        dated = effective
    elif source_dated:
        basis = "source_published_or_updated_at"
        dated = source_dated
    else:
        return TemporalCoverage(
            intent=intent,
            requested_as_of=requested_as_of,
            checked_at=checked_at,
            status="unknown",
            basis="none",
            reason_codes=("temporal_coverage_unknown",),
            statement="时效范围：现有证据未提供可核验的事实生效或来源发布时间，无法确认其覆盖到当前时点。",
            note_required=True,
        )

    dated_ids = tuple(evidence_identity(item) for item, _ in dated)
    supported_through = max(value for _, value in dated)
    complete = len(dated) == len(eligible) and bool(eligible)

    if intent == "latest":
        status: TemporalStatus = "bounded" if complete else "partially_bounded"
        reasons = () if complete else ("temporal_coverage_partial",)
        statement = f"时效范围：可核验证据支持至 {_date_label(supported_through)}；未使用检索时间替代事实时点。"
        return TemporalCoverage(
            intent=intent,
            checked_at=checked_at,
            status=status,
            supported_through=supported_through,
            basis=basis,
            evidence_ids=dated_ids,
            reason_codes=reasons,
            statement=statement,
            note_required=not complete,
        )

    assert requested_as_of is not None
    future = requested_as_of > checked_at
    exact_effective_support = any(
        item.effective_at is not None and item.effective_at <= requested_as_of
        for item in eligible
    )
    if future:
        return TemporalCoverage(
            intent="as_of",
            requested_as_of=requested_as_of,
            checked_at=checked_at,
            status="partially_bounded",
            supported_through=supported_through,
            basis=basis,
            evidence_ids=dated_ids,
            reason_codes=("requested_as_of_in_future",),
            statement=(
                f"时效范围：请求时点 {_date_label(requested_as_of)} 尚未到达；正文仅使用截至 "
                f"{_date_label(supported_through)} 可核验的最近资料。"
            ),
            note_required=True,
        )
    # A source publication/update date only proves when the document existed. It
    # does not prove that the document describes the requested historical state.
    bounded = exact_effective_support
    if bounded:
        return TemporalCoverage(
            intent="as_of",
            requested_as_of=requested_as_of,
            checked_at=checked_at,
            status="bounded" if complete else "partially_bounded",
            supported_through=min(requested_as_of, supported_through),
            basis=basis,
            evidence_ids=dated_ids,
            reason_codes=() if complete else ("temporal_coverage_partial",),
            statement=f"时效范围：正文按 {_date_label(requested_as_of)} 的历史状态组织证据与结论。",
            note_required=not complete,
        )
    return TemporalCoverage(
        intent="as_of",
        requested_as_of=requested_as_of,
        checked_at=checked_at,
        status="partially_bounded",
        supported_through=supported_through,
        basis=basis,
        evidence_ids=dated_ids,
        reason_codes=("requested_as_of_not_covered",),
        statement=(
            f"时效范围：现有证据不能完整还原 {_date_label(requested_as_of)}；正文仅陈述可核验的最近边界。"
        ),
        note_required=True,
    )


def temporal_degraded_reasons(coverage: TemporalCoverage) -> list[str]:
    if coverage.intent == "none" or coverage.status in {"not_requested", "bounded"}:
        return []
    return list(coverage.reason_codes) or ["temporal_coverage_partial"]


def terminal_status(*, timed_out: bool, degraded_reasons: list[str]) -> str:
    """Apply T46 terminal precedence for successful, non-error runs."""

    if timed_out:
        return "partial_timeout"
    if degraded_reasons:
        return "completed_degraded"
    return "completed"


def _parse_date(value: str) -> float | None:
    try:
        return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()
    except ValueError:
        return None


def _date_label(value: float) -> str:
    return datetime.fromtimestamp(float(value), tz=timezone.utc).date().isoformat()
