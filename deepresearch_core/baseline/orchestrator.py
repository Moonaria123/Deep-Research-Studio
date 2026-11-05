"""Research Orchestrator: the four-mode state machine.

Coordinates retrieval, answer composition, council, citations, memory, and the job
store. Every state transition emits a ``node_status`` event (persisted for resume and
forwarded to an optional SSE sink). Cancellation is cooperative between transitions.
"""

from __future__ import annotations

import inspect
import os
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from typing import Callable

from .adapters import Document, LLMAdapter, SearchCallResult, SearchToolAdapter
from .budget import (
    MEMORY_POLICY_DEFAULT,
    Budget,
    Mode,
    RetrievalBudget,
    RunRequest,
    retrieval_budget_for,
    validate_deadline_budget,
)
from .citations import (
    COMPLEMENTARY_EVIDENCE_CHANNEL,
    Citation,
    GroundedAnswer,
    analyze_evidence_pack_conflicts,
    analyze_evidence_conflicts,
    derive_confidence_label,
    enforce_grounding,
    enrich_council_structure,
)
from .composer import AnswerComposer, StreamComposeMeta
from .context_budget import ContextBudgetConfig, coerce_payload, default_config, from_env
from .council import CouncilMemberSpec, build_council_trust_constraints, run_council
from .deep_stages import expand_queries, extract_notes, render_evidence_matrix
from .deep_checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointCodecError,
    CheckpointStateConflict,
    DeepCheckpointV1,
    encode_deep_checkpoint,
)
from .deadline import ProviderCallBudget, use_provider_call_budget
from .errors import BaselineError, ErrorCode
from .evidence_eligibility import (
    EvidenceAssessment,
    EvidenceEligibilityOutcome,
    RemediationOutcome,
    ResearchIntent,
    apply_citation_limit,
    assess_evidence,
    focused_recovery_query,
    output_instruction,
    output_shape_valid,
    parse_research_intent,
    started_remediation,
)
from .evidence_trust import (
    EvidenceConflict,
    RetrievalOutcome,
    TemporalCoverage,
    assess_temporal_coverage,
    detect_temporal_intent,
    evidence_identity,
    provenance_family,
    temporal_degraded_reasons,
    temporal_validation_query,
    terminal_status,
)
from .external_candidates import (
    ExternalCandidate,
    bind_fetched_content,
    fetched_body_may_contain_focused_evidence,
    candidate_from_search_hit,
    select_fetched_evidence_excerpt,
    snippet_evidence,
    title_intent_fit_score,
    title_query_phrase_span,
)
from .llm_call_params import LLMCallParams
from .history import SCOPE_INTRA, ConversationIndex
from .indexing import Indexer
from .knowledge_graph.retrieval import (
    KGRetrievalDiagnostics,
    KnowledgeGraphEvidenceRetriever,
)
from .library_retrieval import LibraryRetriever, render_library_context
from .memory import MemoryEngine, MemoryLayer
from .research_state import ResearchState
from .retrieval import EvidenceItem, Retriever, reciprocal_rank_fusion
from .search_options import SearchOptions, freeze_search_locale, runtime_search_options
from .store import Store
from ..observability import get_logger, span
from ..standalone.llm import redact_text

EventSink = Callable[[dict], None]
LLMResolver = Callable[[RunRequest], "LLMAdapter | None"]
CouncilResolver = Callable[[RunRequest], "list[CouncilMemberSpec] | list[LLMAdapter]"]
ParamsResolver = Callable[[RunRequest], LLMCallParams]
PostRunArtifactHook = Callable[[RunRequest, "RunOutcome"], None]
PostContinuityHook = Callable[
    [RunRequest, "RunOutcome", str, dict[str, list[str]]],
    None,
]

# Mode-aware grounding thresholds (fast trades depth for speed).
_MIN_COVERAGE = {Mode.FAST: 0.50, Mode.MULTIPLE: 0.70, Mode.DEEP: 0.85, Mode.COUNCIL: 0.85}
_SECRET_WARNING_RE = re.compile(r"(sk-|ghp_|AKIA|xoxb-|xoxp-|BEGIN .*PRIVATE KEY)", re.IGNORECASE)
_STREAM_FLUSH_INTERVAL_S = 0.05
_STREAM_FLUSH_BYTES = 256
_STREAM_EVENT_MAX_BYTES = 16 * 1024
_STREAM_EVENT_MAX_COUNT = 4096
_STREAM_DRAFT_MAX_BYTES = 4 * 1024 * 1024
_DEEP_SECTION_WORK_RESERVE_S = 15.0
_COMPLEMENTARY_EVIDENCE_MAX_CHARS = 4096
_COMPLEMENTARY_EVIDENCE_PIECE_CHARS = 600
_KG_DEGRADED_REASON_CODES = frozenset(
    {
        "graph_not_ready",
        "snapshot_read_failed",
        "snapshot_signature_mismatch",
        "snapshot_signature_unavailable",
        "snapshot_unavailable",
        "snapshot_version_changed",
        "workspace_mismatch",
        "kg_read_failed",
    }
)
_DEEP_SECTION_SCOPE_STOPWORDS = frozenset(
    {
        "and",
        "a",
        "an",
        "answer",
        "authoritative",
        "capabilities",
        "concise",
        "documentation",
        "explain",
        "explaining",
        "for",
        "in",
        "its",
        "match",
        "of",
        "query",
        "ranking",
        "report",
        "research",
        "search",
        "syntax",
        "the",
        "to",
        "using",
        "what",
        "write",
    }
)
_DEEP_SECTION_PLAN_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "authoritative",
        "concise",
        "documentation",
        "explain",
        "explaining",
        "for",
        "in",
        "its",
        "of",
        "report",
        "research",
        "the",
        "to",
        "using",
        "with",
        "write",
    }
)
_LOG = get_logger("deepresearch.orchestrator")


def _deep_section_work_reserve_s(
    timeout_s: float,
    retrieval: RetrievalBudget,
) -> float:
    """Reserve section processing only from budget beyond one full provider call."""

    surplus_s = float(timeout_s) - retrieval.reserve_total_s - 60.0
    return min(_DEEP_SECTION_WORK_RESERVE_S, max(0.0, surplus_s))


def _deep_section_round_limit(intent: ResearchIntent, budget: Budget) -> int:
    """Honor an explicit short answer shape before the Deep mode default breadth."""

    configured = max(1, int(budget.max_searches))
    if intent.output_shape != "sentence_count" or not intent.sentence_count:
        return configured
    sentences = max(1, int(intent.sentence_count))
    return min(configured, max(1, (sentences + 1) // 2))


def _deep_plan_terms(text: str) -> set[str]:
    return {
        token.casefold().rstrip(".")
        for token in re.findall(r"[A-Za-z][A-Za-z0-9.+#-]*|[\u3400-\u9fff]{2,}", text)
        if len(token) > 2 and token.casefold().rstrip(".") not in _DEEP_SECTION_PLAN_STOPWORDS
    }


def _select_deep_sections(sections: list[str], focus: str, count: int) -> list[str]:
    """Keep the opening scope plus sections that add the most requested-topic coverage."""

    limit = max(1, int(count))
    if len(sections) <= limit:
        return list(sections)

    focus_terms = _deep_plan_terms(focus)
    selected = [0]
    covered = _deep_plan_terms(sections[0]) & focus_terms
    remaining = set(range(1, len(sections)))
    while remaining and len(selected) < limit:
        index = max(
            remaining,
            key=lambda item: (
                len((_deep_plan_terms(sections[item]) & focus_terms) - covered),
                len(_deep_plan_terms(sections[item]) & focus_terms),
                -item,
            ),
        )
        selected.append(index)
        covered.update(_deep_plan_terms(sections[index]) & focus_terms)
        remaining.remove(index)
    return [sections[index] for index in sorted(selected)]


def _deep_section_focus(parent_focus: str, section_focus: str) -> str:
    """Keep the run entity scope when evaluating a generated Deep section."""

    parent = " ".join(str(parent_focus or "").split()).strip()
    section = " ".join(str(section_focus or "").split()).strip()
    if not parent:
        return section
    if not section or section.casefold() in parent.casefold():
        return parent
    return f"{parent} {section}"


def _deep_section_search_focus(parent_focus: str, section_focus: str) -> str:
    """Prefix a section query with stable subject identity, not every parent subtopic."""

    parent = " ".join(str(parent_focus or "").split()).strip()
    section = " ".join(str(section_focus or "").split()).strip()
    if not parent:
        return section
    if not section:
        return parent

    section_tokens = {
        token.casefold().rstrip(".")
        for token in re.findall(r"[A-Za-z][A-Za-z0-9.+#-]*|[\u3400-\u9fff]{2,}", section)
    }
    identity: list[str] = []
    identity_seen: set[str] = set()
    for token in re.findall(r"[A-Za-z][A-Za-z0-9.+#-]*|[\u3400-\u9fff]{2,}", parent):
        token = token.rstrip(".")
        normalized = token.casefold()
        if (
            len(normalized) < 2
            or normalized in _DEEP_SECTION_SCOPE_STOPWORDS
            or normalized in section_tokens
        ):
            continue
        identity_like = bool(
            any(char.isdigit() for char in token)
            or token[:1].isupper()
            or re.fullmatch(r"[\u3400-\u9fff]{2,}", token)
            or not identity
        )
        if identity_like and normalized not in identity_seen:
            identity.append(token)
            identity_seen.add(normalized)
        if len(identity) >= 4:
            break
    return " ".join([*identity, section]).strip() if identity else section


def _retrieval_intent(
    intent: ResearchIntent,
    relevance_query: str | None,
) -> ResearchIntent:
    """Project a scoped retrieval focus without changing run-level source rules."""

    focus = " ".join(str(relevance_query or "").split()).strip()
    if not focus or focus == intent.research_focus:
        return intent
    return replace(
        intent,
        research_focus=focus,
        citation_limit=None,
        output_shape="default",
        sentence_count=None,
        max_length=None,
    )


def _safe_warning(message: str) -> str:
    """Return a bounded warning message without paths or secret-like values."""

    text = re.sub(r"[A-Za-z]:\\[^\s]+", "[path]", str(message or ""))
    text = re.sub(r"/[^\s]+", "[path]", text)
    text = _SECRET_WARNING_RE.sub("[redacted]", text)
    return text[:240]


@dataclass
class _RunDeadline:
    """Absolute request deadline represented on the worker's monotonic clock."""

    timeout_s: float
    deadline_at: float = 0.0
    clock: Callable[[], float] = time.monotonic
    wall_clock: Callable[[], float] = time.time
    started_at: float = field(init=False)
    worker_started_at: float = field(init=False)
    monotonic_deadline: float = field(init=False)

    def __post_init__(self) -> None:
        self.started_at = float(self.clock())
        self.worker_started_at = float(self.wall_clock())
        if self.deadline_at > 0:
            remaining = max(0.0, float(self.deadline_at) - self.worker_started_at)
        else:
            remaining = max(0.0, float(self.timeout_s))
            self.deadline_at = self.worker_started_at + remaining
        self.monotonic_deadline = self.started_at + remaining

    def remaining_s(self, *, reserve_s: float = 0.0) -> float:
        """Return usable time after an optional downstream reserve."""

        return max(
            0.0,
            self.monotonic_deadline - float(self.clock()) - max(0.0, float(reserve_s)),
        )

    def expired(self, *, reserve_s: float = 0.0) -> bool:
        """Return whether the absolute deadline or phase reserve is exhausted."""

        return self.remaining_s(reserve_s=reserve_s) <= 0

    def provider_budget(
        self,
        *,
        phase_reserve_s: float = 0.0,
        retrieval_budget: RetrievalBudget | None = None,
        interrupt_check: Callable[[], None] | None = None,
        max_attempts: int = 1,
    ) -> ProviderCallBudget:
        """Create a provider budget whose phase deadline preserves later work."""

        phase_deadline = self.monotonic_deadline - max(0.0, float(phase_reserve_s))
        return ProviderCallBudget(
            request_deadline=self.monotonic_deadline,
            phase_deadline=phase_deadline,
            writer_reserve_s=(retrieval_budget.writer_reserve_s if retrieval_budget else 0.0),
            finalize_reserve_s=(retrieval_budget.finalize_reserve_s if retrieval_budget else 0.0),
            cleanup_reserve_s=(retrieval_budget.cleanup_reserve_s if retrieval_budget else 0.0),
            max_attempts=max(1, int(max_attempts)),
            cancel_or_timeout_check=interrupt_check,
            clock=self.clock,
        )


@dataclass
class _AnswerStream:
    """Bounded, persisted answer-delta coalescer for one run."""

    emit: Callable[[str, dict], None]
    cancel_check: Callable[[], None]
    requested_at: float
    metrics_sink: Callable[[dict[str, object]], None] | None = field(default=None, repr=False)
    generation_number: int = 0
    generation_id: str | None = None
    delta_seq: int = 0
    delta_event_count: int = 0
    delta_bytes: int = 0
    draft_bytes: int = 0
    first_answer_delta_ms: float | None = None
    sync_fallback_used: bool = False
    _buffer: str = ""
    _last_flush_at: float = field(default_factory=time.monotonic)
    _draft_active: bool = False
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _flush_timer: threading.Timer | None = field(default=None, repr=False)
    _flush_timer_token: int = field(default=0, repr=False)
    _timer_error: Exception | None = field(default=None, repr=False)

    def accept(self, delta: str) -> None:
        """Accept one provider chunk and persist it within the coalescing bounds."""

        if not delta:
            return
        self.cancel_check()
        with self._lock:
            self._raise_timer_error_locked()
            if self.generation_id is None:
                self.generation_number += 1
                self.generation_id = f"g{self.generation_number}"
                self.delta_seq = 0
                self._last_flush_at = time.monotonic()
            incoming_bytes = len(delta.encode("utf-8"))
            if self.draft_bytes + incoming_bytes > _STREAM_DRAFT_MAX_BYTES:
                self.reset("draft_invalidated")
                raise BaselineError(
                    ErrorCode.STREAM_LIMIT_EXCEEDED,
                    "Streamed answer exceeded the per-run draft limit.",
                )
            self.draft_bytes += incoming_bytes
            if self.delta_seq == 0 and not self._buffer:
                self._emit_text(delta)
                return
            self._buffer += delta
            elapsed = time.monotonic() - self._last_flush_at
            if len(self._buffer.encode("utf-8")) >= _STREAM_FLUSH_BYTES or elapsed >= _STREAM_FLUSH_INTERVAL_S:
                self._cancel_flush_timer_locked()
                self._flush_buffer_locked()
            else:
                self._schedule_flush_locked(_STREAM_FLUSH_INTERVAL_S - elapsed)

    def flush(self) -> None:
        """Persist the current coalesced buffer, split at the event byte bound."""

        with self._lock:
            self._cancel_flush_timer_locked()
            self._raise_timer_error_locked()
            self._flush_buffer_locked()

    def finish_generation(self) -> None:
        self.flush()

    def reset(self, reason: str) -> None:
        """Invalidate the visible draft and make the next delta start a new generation."""

        with self._lock:
            self._cancel_flush_timer_locked()
            self._timer_error = None
            generation_id = self.generation_id
            self._buffer = ""
            self._draft_active = False
            self.generation_id = None
            self.delta_seq = 0
            if generation_id is not None:
                self.emit("answer_reset", {"generation_id": generation_id, "reason": reason})

    def commit(self) -> None:
        """Mark the atomically persisted run result as authoritative."""

        with self._lock:
            self._cancel_flush_timer_locked()
            self._buffer = ""
            self._draft_active = False

    def mark_sync_fallback(self, used: bool) -> None:
        with self._lock:
            changed = bool(used) and not self.sync_fallback_used
            self.sync_fallback_used = self.sync_fallback_used or bool(used)
            if changed:
                self._persist_metrics_locked()

    def metrics(self) -> dict[str, object]:
        with self._lock:
            return self._metrics_locked()

    @property
    def draft_active(self) -> bool:
        with self._lock:
            return self._draft_active or self.generation_id is not None

    def _flush_buffer_locked(self) -> None:
        if not self._buffer:
            return
        buffered, self._buffer = self._buffer, ""
        self._emit_text(buffered)

    def _schedule_flush_locked(self, delay_s: float) -> None:
        if self._flush_timer is not None:
            return
        self._flush_timer_token += 1
        token = self._flush_timer_token
        timer = threading.Timer(max(0.001, delay_s), self._flush_from_timer, args=(token,))
        timer.daemon = True
        self._flush_timer = timer
        timer.start()

    def _flush_from_timer(self, token: int) -> None:
        with self._lock:
            if token != self._flush_timer_token:
                return
            self._flush_timer = None
            if self._timer_error is not None:
                return
            try:
                self._flush_buffer_locked()
            except Exception as error:  # noqa: BLE001 - replayed on the producer thread
                self._timer_error = error

    def _cancel_flush_timer_locked(self) -> None:
        timer, self._flush_timer = self._flush_timer, None
        if timer is not None:
            self._flush_timer_token += 1
            timer.cancel()

    def _raise_timer_error_locked(self) -> None:
        if self._timer_error is None:
            return
        error, self._timer_error = self._timer_error, None
        raise error

    def _emit_text(self, text: str) -> None:
        for piece in _split_utf8(text, _STREAM_EVENT_MAX_BYTES):
            if self.delta_event_count >= _STREAM_EVENT_MAX_COUNT:
                self.reset("draft_invalidated")
                raise BaselineError(
                    ErrorCode.STREAM_LIMIT_EXCEEDED,
                    "Streamed answer exceeded the per-run event limit.",
                )
            self.cancel_check()
            self.delta_seq += 1
            self.emit(
                "answer_delta",
                {
                    "delta": piece,
                    "delta_seq": self.delta_seq,
                    "generation_id": self.generation_id,
                },
            )
            piece_bytes = len(piece.encode("utf-8"))
            self.delta_event_count += 1
            self.delta_bytes += piece_bytes
            self._draft_active = True
            self._last_flush_at = time.monotonic()
            if self.first_answer_delta_ms is None:
                self.first_answer_delta_ms = max(0.0, (time.time() - self.requested_at) * 1000.0)
                self._persist_metrics_locked()

    def _metrics_locked(self) -> dict[str, object]:
        return {
            "first_answer_delta_ms": self.first_answer_delta_ms,
            "delta_event_count": self.delta_event_count,
            "delta_bytes": self.delta_bytes,
            "sync_fallback_used": self.sync_fallback_used,
        }

    def _persist_metrics_locked(self) -> None:
        if self.metrics_sink is not None:
            self.metrics_sink(self._metrics_locked())


def _split_utf8(text: str, max_bytes: int) -> list[str]:
    """Split text without breaking a code point while respecting a UTF-8 byte cap."""

    if len(text.encode("utf-8")) <= max_bytes:
        return [text]
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for char in text:
        char_size = len(char.encode("utf-8"))
        if current and size + char_size > max_bytes:
            parts.append("".join(current))
            current = []
            size = 0
        current.append(char)
        size += char_size
    if current:
        parts.append("".join(current))
    return parts


@dataclass
class _RunContext:
    """Per-run resolved LLM, composer, and council members (thread-local to a run)."""

    llm: LLMAdapter
    composer: AnswerComposer
    members: "list[CouncilMemberSpec] | list[LLMAdapter]"
    llm_params: LLMCallParams = field(default_factory=LLMCallParams)
    conversation_context: str = ""
    llm_model_refs: tuple[str, ...] = ()
    search_providers: set[str] = field(default_factory=set)
    provider_circuit_open: set[str] = field(default_factory=set)
    providers_attempted: list[str] = field(default_factory=list)
    providers_skipped: list[str] = field(default_factory=list)
    provider_failure_reasons: dict[str, str] = field(default_factory=dict)
    retrieval_outcomes: list[RetrievalOutcome] = field(default_factory=list)
    degraded_reasons: list[str] = field(default_factory=list)
    answer_stream: _AnswerStream | None = None
    research_intent: ResearchIntent | None = None
    evidence_eligibility: dict[str, EvidenceEligibilityOutcome] = field(default_factory=dict)
    remediation: RemediationOutcome = field(default_factory=RemediationOutcome)
    recovery_searches_max: int = 1
    recovery_searches_used: int = 0
    ordinary_searches_used: int = 0
    temporal_reason_codes: set[str] = field(default_factory=set)
    retrieval_budget: RetrievalBudget | None = None
    phase_metrics: list[dict[str, object]] = field(default_factory=list)
    writer_started_at: float | None = None
    retrieval_rejection_reason_codes: set[str] = field(default_factory=set)
    grounded_partial_answer: GroundedAnswer | None = None
    grounded_partial_evidence: list[EvidenceItem] = field(default_factory=list)
    grounded_partial_extra: dict[str, object] = field(default_factory=dict)
    prepared_external_documents: dict[str, tuple[object, int]] = field(default_factory=dict)
    unready_external_provenance_roots: set[str] = field(default_factory=set)
    final_adopted_evidence: list[EvidenceItem] = field(default_factory=list)
    writer_diagnostics: dict[str, object] = field(default_factory=dict)
    verification_future: Future | None = None
    verification_executor: ThreadPoolExecutor | None = None
    verification_started_at: float = 0.0
    verification_deadline: float = 0.0
    verification_reasons: list[str] = field(default_factory=list)
    regeneration_evidence: list[EvidenceItem] = field(default_factory=list)
    fresh_search_required: bool = False
    kg_retrieval: KGRetrievalDiagnostics = field(
        default_factory=lambda: KGRetrievalDiagnostics(requested=False)
    )


@dataclass
class _ExternalSearchHandle:
    """Run-scoped pure-network search started before local Store-backed lanes."""

    result: SearchCallResult | None = None
    future: Future | None = None
    executor: ThreadPoolExecutor | None = None
    provider_circuit: set[str] = field(default_factory=set)
    started_at: float = 0.0
    remaining_before_s: float | None = None


@dataclass
class RunOutcome:
    """The result of a completed run."""

    run_id: str
    mode: str
    answer: GroundedAnswer
    status: str = "completed"
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-serializable view."""

        return {
            "run_id": self.run_id,
            "mode": self.mode,
            "status": self.status,
            "answer": self.answer.to_dict(),
            **self.extra,
        }


class DeepRunPaused(RuntimeError):
    """Internal control-flow signal after a paused checkpoint is durably committed."""


class Orchestrator:
    """Executes research runs across the four modes."""

    def __init__(
        self,
        *,
        store: Store,
        llm: LLMAdapter,
        retriever: Retriever | None = None,
        indexer: Indexer | None = None,
        search: SearchToolAdapter | None = None,
        memory: MemoryEngine | None = None,
        council_members: list[LLMAdapter] | None = None,
        enforce_grounding_gate: bool = True,
        llm_resolver: LLMResolver | None = None,
        council_resolver: CouncilResolver | None = None,
        params_resolver: ParamsResolver | None = None,
        history: ConversationIndex | None = None,
        context_budget: ContextBudgetConfig | None = None,
        post_run_artifacts: PostRunArtifactHook | None = None,
        post_continuity_persisted: PostContinuityHook | None = None,
        library_retriever: LibraryRetriever | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._store = store
        self._llm = llm
        self._retriever = retriever or Retriever(store)
        self._indexer = indexer or (Indexer(store) if search is not None else None)
        self._search = search
        self._composer = AnswerComposer(llm)
        self._memory = memory
        self._council_members = council_members or []
        self._enforce = enforce_grounding_gate
        self._llm_resolver = llm_resolver
        self._council_resolver = council_resolver
        self._params_resolver = params_resolver
        self._history = history or ConversationIndex(store)
        # Instance-level fallback budget (used only when a run carries no resolved
        # override; the API resolves and embeds the effective config per run).
        self._context_budget = context_budget
        self._post_run_artifacts = post_run_artifacts
        self._post_continuity_persisted = post_continuity_persisted
        self._library_retriever = library_retriever
        self._kg_retriever = KnowledgeGraphEvidenceRetriever(store)
        self._clock = clock or time.monotonic
        self._cancelled: set[str] = set()
        self._lock = threading.Lock()

    # --- public API -----------------------------------------------------------

    def request_cancel(self, run_id: str) -> None:
        """Cooperatively cancel a run."""

        if self._store.transition_run(
            run_id,
            expected_statuses={"queued", "running", "pausing"},
            status="cancelling",
        ):
            with self._lock:
                self._cancelled.add(run_id)

    def run(self, request: RunRequest, *, run_id: str | None = None, sink: EventSink | None = None) -> RunOutcome:
        """Execute a run end-to-end, persisting events and result."""

        request = self._freeze_run_locale(request)
        budget = request.resolved_budget()
        retrieval_budget = retrieval_budget_for(request.mode)
        validate_deadline_budget(budget, retrieval_budget)
        rid = run_id or self._store.create_run(
            session_id=request.session_id, mode=request.mode.value, query=request.query,
            budget=budget.to_dict(),
            metadata=self._run_metadata(request),
        )
        deadline = _RunDeadline(
            timeout_s=budget.timeout_s,
            deadline_at=request.deadline_at,
            clock=self._clock,
        )
        if not self._store.claim_queued_run(rid):
            current = self._store.get_run(rid) or {}
            with self._lock:
                locally_cancelled = rid in self._cancelled
            if locally_cancelled or current.get("status") == "cancelling":
                error = BaselineError(ErrorCode.RUN_CANCELLED, "run was cancelled")
                diagnostics = self._runtime_diagnostics(
                    request=request,
                    deadline=deadline,
                    ctx=None,
                    terminal_at=time.time(),
                    deadline_met=False,
                )
                if self._store.transition_run(
                    rid,
                    expected_statuses={"cancelling"},
                    status="cancelled",
                    error=error.to_payload(),
                    metadata_patch=diagnostics,
                ):
                    self._emit(rid, "error", error.to_payload(), sink)
                raise error
            raise BaselineError(
                ErrorCode.INTERNAL_ERROR,
                "run could not be claimed for execution",
                details={"run_id": rid, "status": current.get("status")},
            )
        ctx: _RunContext | None = None
        try:
            # Queue time is part of the request budget.  Check it before resolving a
            # model, continuity embeddings, or any other provider-backed context.
            self._check_cancel(rid)
            self._check_deadline(deadline)
            context_started = time.time()
            with use_provider_call_budget(
                deadline.provider_budget(
                    retrieval_budget=retrieval_budget,
                    interrupt_check=self._stream_cancel_checker(rid, deadline),
                )
            ):
                ctx = self._build_context(request)
            ctx.retrieval_budget = retrieval_budget
            ctx.recovery_searches_max = 0 if request.mode is Mode.COUNCIL else 1
            ctx.research_intent = parse_research_intent(request.query)
            regeneration = (self._store.get_run(rid) or {}).get("metadata", {}).get("regeneration_context", {})
            ctx.fresh_search_required = bool(regeneration.get("fresh_search_required"))
            if ctx.fresh_search_required and regeneration.get("fresh_search_available") is False:
                ctx.verification_reasons.append("external_search_unavailable")
            for index, item in enumerate(regeneration.get("seed_evidence", [])[:32]):
                text = str(item.get("snippet") or item.get("text") or "")
                artifact_id = str(item.get("artifact_id") or "")
                artifact_type = str(item.get("artifact_type") or "web_document")
                if artifact_type == "memory" and (not artifact_id or self._store.get_memory(artifact_id) is None):
                    continue
                if artifact_type == "library_document" and (not artifact_id or self._store.get_document(artifact_id) is None):
                    continue
                if text and item.get("grounding_eligible", True):
                    ctx.regeneration_evidence.append(EvidenceItem(
                        chunk_id=int(item.get("chunk_id") or 0), document_id=str(item.get("document_id") or ""),
                        url=str(item.get("url") or ""), title=str(item.get("title") or ""),
                        text=text, channels=["prior_report"], origin=str(item.get("origin") or "external"),
                        score=float(item.get("score") or 0), artifact_type=artifact_type, artifact_id=artifact_id,
                        source_status=str(item.get("source_status") or "active"),
                        published_at=item.get("published_at"), updated_at=item.get("updated_at"), effective_at=item.get("effective_at"),
                        provenance_refs=list(item.get("provenance_refs") or []),
                        retrieved_at=item.get("retrieved_at"), search_provider=str(item.get("search_provider") or ""),
                        fetch_provider=str(item.get("fetch_provider") or ""),
                    ))
            ctx.kg_retrieval = self._initial_kg_retrieval_diagnostics(request)
            for reason in ctx.kg_retrieval.reason_codes:
                if reason in _KG_DEGRADED_REASON_CODES:
                    self._add_degraded_reason(
                        ctx,
                        f"kg_retrieval_{reason}",
                    )
            ctx.answer_stream = _AnswerStream(
                emit=lambda event_type, payload: self._emit(rid, event_type, payload, sink),
                cancel_check=self._stream_cancel_checker(rid, deadline),
                requested_at=request.requested_at,
                metrics_sink=lambda metrics: self._store.update_run(
                    rid,
                    metadata_patch={"stream_metrics": metrics},
                ),
            )
            self._append_phase_metric(
                ctx,
                stage="context_resolution",
                started_at=context_started,
                finished_at=time.time(),
                status="completed",
                deadline=deadline,
            )
            with span("orchestrator.run", run_id=rid, mode=request.mode.value):
                with use_provider_call_budget(
                    deadline.provider_budget(
                        retrieval_budget=retrieval_budget,
                        interrupt_check=self._stream_cancel_checker(rid, deadline),
                    )
                ):
                    outcome = self._dispatch(request, budget, rid, sink, ctx, deadline)
            ctx.kg_retrieval.mark_adopted(ctx.final_adopted_evidence)
            outcome.extra["execution_provenance"] = self._execution_provenance(ctx)
            outcome.extra.setdefault(
                "retrieval_outcomes",
                [item.to_dict() for item in ctx.retrieval_outcomes],
            )
            outcome.extra.setdefault("degraded_reasons", list(dict.fromkeys(ctx.degraded_reasons)))
            outcome.extra.setdefault("research_intent", ctx.research_intent.to_dict())
            outcome.extra.setdefault(
                "evidence_eligibility",
                [item.to_dict() for item in ctx.evidence_eligibility.values()],
            )
            outcome.extra.setdefault("remediation", ctx.remediation.to_dict())
            outcome.extra.setdefault("recovery_searches_max", ctx.recovery_searches_max)
            outcome.extra.setdefault("recovery_searches_used", ctx.recovery_searches_used)
            outcome.extra.setdefault("resolved_search_usage", dict(request.resolved_search_usage))
            outcome.extra["kg_retrieval"] = ctx.kg_retrieval.to_dict()
            stream_metrics = self._stream_metrics(ctx, request)
            outcome.extra["stream_metrics"] = stream_metrics
            outcome.extra["verification"] = {
                "status": "running", "started_at": ctx.verification_started_at or time.time(),
            }
            outcome.answer.confidence_label = ""
            final_status = outcome.status or "completed"
            self._check_cancel(rid)
            if deadline.expired() and final_status in {"completed", "completed_degraded"}:
                final_status = "partial_timeout"
                outcome.status = final_status
                outcome.extra["completion_reason"] = "timeout"
            diagnostics = self._runtime_diagnostics(
                request=request,
                deadline=deadline,
                ctx=ctx,
                terminal_at=time.time(),
                deadline_met=final_status not in {"error", "cancelled", "partial_timeout"},
            )
            outcome.extra.update(diagnostics)
            transitioned = self._store.transition_run(
                rid,
                expected_statuses={"running"},
                status=final_status,
                result=outcome.to_dict(),
                metadata_patch={"stream_metrics": stream_metrics, **diagnostics},
                citation_rows=self._citation_rows(outcome.answer.citations),
            )
            if not transitioned:
                raise BaselineError(
                    ErrorCode.RUN_CANCELLED,
                    "run reached a terminal state before finalization completed",
                )
            self._promote_final_external_evidence(ctx.final_adopted_evidence)
            ctx.answer_stream.commit()
            terminal_payload = {"status": final_status}
            if final_status == "partial_timeout":
                terminal_payload["code"] = ErrorCode.RUN_TIMEOUT.value
            self._emit(rid, "workflow_end", terminal_payload, sink)
            # Raw continuity is committed after the terminal event and before any
            # semantic maintenance enqueue. Failures remain out-of-band: the run is
            # already terminal and no late SSE event may be appended.
            persisted_rows: dict[str, list[str]] = {
                "session_memory_ids": [],
                "message_ids": [],
            }
            try:
                memory_id = self._maybe_remember(request, outcome)
                if memory_id:
                    persisted_rows["session_memory_ids"].append(memory_id)
            except Exception as error:  # noqa: BLE001 - run already succeeded
                _LOG.warning(
                    "post_continuity_memory_persist_failed",
                    run_id=rid,
                    error_type=error.__class__.__name__,
                )
            try:
                persisted_rows["message_ids"].extend(
                    self._persist_messages(request, outcome, rid)
                )
            except Exception as error:  # noqa: BLE001 - run already succeeded
                _LOG.warning(
                    "post_continuity_message_persist_failed",
                    run_id=rid,
                    error_type=error.__class__.__name__,
                )
            if any(persisted_rows.values()) and self._post_continuity_persisted is not None:
                try:
                    self._post_continuity_persisted(
                        request,
                        outcome,
                        rid,
                        persisted_rows,
                    )
                except Exception as error:  # noqa: BLE001 - maintenance is best-effort
                    _LOG.warning(
                        "semantic_enrichment_enqueue_failed",
                        run_id=rid,
                        error_type=error.__class__.__name__,
                    )
            try:
                if self._post_run_artifacts is not None:
                    self._post_run_artifacts(request, outcome)
            except Exception as error:  # noqa: BLE001 - run already succeeded
                self._emit(
                    rid,
                    "library_artifact_warning",
                    {"stage": "post_run_artifacts", "message": _safe_warning(str(error) or error.__class__.__name__)},
                    sink,
                )
            self._complete_verification(request, outcome, ctx)
            return outcome
        except DeepRunPaused:
            # The safe-boundary transaction already persisted ``paused`` and its
            # terminal SSE event. No result, citations, continuity, or artifacts
            # may be written for the paused parent.
            raise
        except BaselineError as error:
            # A cooperative cancel is a terminal "cancelled" state, not a failure.
            # This is the cross-process exit path too: the worker observes the
            # persisted ``cancelling`` status in ``_check_cancel`` and lands here.
            current = self._store.get_run(rid) or {}
            if error.code is ErrorCode.RUN_TIMEOUT and str(current.get("status") or "") == "cancelling":
                error = BaselineError(ErrorCode.RUN_CANCELLED, "run was cancelled")
            if (
                error.code is ErrorCode.RUN_TIMEOUT
                and ctx is not None
                and ctx.grounded_partial_answer is not None
                and str(current.get("status") or "") == "running"
            ):
                return self._complete_grounded_partial_timeout(
                    request=request,
                    rid=rid,
                    sink=sink,
                    ctx=ctx,
                    deadline=deadline,
                )
            final_status = "cancelled" if error.code is ErrorCode.RUN_CANCELLED else "error"
            stream = ctx.answer_stream if ctx is not None else None
            may_finalize = str(current.get("status") or "") in {"running", "cancelling"}
            if stream is not None and stream.draft_active and may_finalize:
                stream.reset(self._reset_reason_for_error(error))
            diagnostics = self._runtime_diagnostics(
                request=request,
                deadline=deadline,
                ctx=ctx,
                terminal_at=time.time(),
                deadline_met=False,
            )
            metadata_patch = {
                "stream_metrics": self._stream_metrics(ctx, request),
                **diagnostics,
            }
            if ctx is not None:
                metadata_patch.update(self._t48_audit_metadata(ctx))
            transitioned = self._store.transition_run(
                rid,
                expected_statuses={"running", "cancelling"},
                status=final_status,
                error=error.to_payload(),
                metadata_patch=metadata_patch,
            )
            if transitioned:
                self._emit(rid, "error", error.to_payload(), sink)
            raise
        except Exception as error:  # noqa: BLE001 - normalize to safe internal error
            payload = BaselineError(ErrorCode.INTERNAL_ERROR, "run failed").to_payload()
            stream = ctx.answer_stream if ctx is not None else None
            current = self._store.get_run(rid) or {}
            may_finalize = str(current.get("status") or "") in {"running", "cancelling"}
            if stream is not None and stream.draft_active and may_finalize:
                stream.reset("draft_invalidated")
            diagnostics = self._runtime_diagnostics(
                request=request,
                deadline=deadline,
                ctx=ctx,
                terminal_at=time.time(),
                deadline_met=False,
            )
            metadata_patch = {
                "stream_metrics": self._stream_metrics(ctx, request),
                **diagnostics,
            }
            if ctx is not None:
                metadata_patch.update(self._t48_audit_metadata(ctx))
            transitioned = self._store.transition_run(
                rid,
                expected_statuses={"running", "cancelling"},
                status="error",
                error=payload,
                metadata_patch=metadata_patch,
            )
            if transitioned:
                self._emit(rid, "error", payload, sink)
            raise BaselineError(ErrorCode.INTERNAL_ERROR, "run failed") from error
        finally:
            if ctx is not None and ctx.verification_executor is not None:
                ctx.verification_executor.shutdown(wait=False, cancel_futures=True)

    def _freeze_run_locale(self, request: RunRequest) -> RunRequest:
        """Freeze search defaults once so worker configuration cannot change the run."""

        settings = getattr(getattr(self._search, "_client", None), "settings", None)
        _, requested, effective = freeze_search_locale(request.search_options, settings)
        requested_at = request.requested_at if request.requested_at > 0 else time.time()
        deadline_at = (
            request.deadline_at
            if request.deadline_at > 0
            # ``requested_at`` existed before T49 as a streaming metric.  A direct
            # legacy caller may supply an old fixture timestamp without an absolute
            # deadline; preserve worker-start timeout semantics for that payload.
            else time.time() + float(request.resolved_budget().timeout_s)
        )
        return replace(
            request,
            search_options=request.search_options,
            requested_locale=dict(request.requested_locale or requested),
            effective_locale=dict(request.effective_locale or effective),
            requested_at=requested_at,
            deadline_at=deadline_at,
        )

    @staticmethod
    def _run_metadata(request: RunRequest) -> dict[str, object]:
        metadata: dict[str, object] = {
            "requested_locale": dict(request.requested_locale),
            "effective_locale": dict(request.effective_locale),
            "resolved_search_usage": dict(request.resolved_search_usage),
            "requested_at": request.requested_at,
            "deadline_at": request.deadline_at,
            "kg_aware_retrieval_enabled": request.kg_aware_retrieval_enabled,
            "kg_retrieval_snapshot": dict(request.kg_retrieval_snapshot),
        }
        if request.search_options is not None:
            metadata["search_options"] = request.search_options.to_dict()
        return metadata

    @staticmethod
    def _initial_kg_retrieval_diagnostics(
        request: RunRequest,
    ) -> KGRetrievalDiagnostics:
        snapshot = (
            request.kg_retrieval_snapshot
            if isinstance(request.kg_retrieval_snapshot, dict)
            else {}
        )
        requested = bool(request.kg_aware_retrieval_enabled)
        effective = bool(snapshot.get("effective", False))
        reason_codes = [
            str(item)
            for item in (snapshot.get("reason_codes") or [])
            if str(item)
        ]
        if requested and not effective and not reason_codes:
            reason_codes = [
                "mode_not_supported"
                if request.mode is Mode.FAST
                else "snapshot_unavailable"
            ]
        diagnostics = KGRetrievalDiagnostics(
            requested=requested,
            effective=effective,
            snapshot_version_id=(
                str(snapshot.get("version_id"))
                if snapshot.get("version_id")
                else None
            ),
            wiki_snapshot_id=(
                str(snapshot.get("wiki_snapshot_id"))
                if snapshot.get("wiki_snapshot_id")
                else None
            ),
            reason_codes=reason_codes,
        )
        return diagnostics

    @staticmethod
    def _reset_reason_for_error(error: BaselineError) -> str:
        if error.code is ErrorCode.RUN_CANCELLED:
            return "cancelled"
        if error.code is ErrorCode.RUN_TIMEOUT:
            return "deadline_exceeded"
        return "draft_invalidated"

    @staticmethod
    def _stream_metrics(ctx: _RunContext | None, request: RunRequest) -> dict[str, object]:
        if ctx is None or ctx.answer_stream is None:
            return {
                "first_answer_delta_ms": None,
                "pre_writer_ms": None,
                "provider_first_delta_ms": None,
                "delta_event_count": 0,
                "delta_bytes": 0,
                "sync_fallback_used": False,
            }
        metrics = dict(ctx.answer_stream.metrics())
        pre_writer_ms = (
            max(0.0, (ctx.writer_started_at - request.requested_at) * 1000.0)
            if ctx.writer_started_at is not None and request.requested_at > 0
            else None
        )
        first_delta_ms = metrics.get("first_answer_delta_ms")
        provider_first_delta_ms = (
            max(0.0, float(first_delta_ms) - pre_writer_ms)
            if first_delta_ms is not None and pre_writer_ms is not None
            else None
        )
        metrics.update(
            {
                "pre_writer_ms": pre_writer_ms,
                "provider_first_delta_ms": provider_first_delta_ms,
            }
        )
        return metrics

    @staticmethod
    def _append_phase_metric(
        ctx: _RunContext,
        *,
        stage: str,
        started_at: float,
        finished_at: float,
        status: str,
        deadline: _RunDeadline,
        plan_id: str | None = None,
        lane: str | None = None,
        query: str | None = None,
        provider: str | None = None,
        attempt: int = 0,
        evidence_id: str | None = None,
        candidate_id: str | None = None,
        remaining_before_s: float | None = None,
    ) -> None:
        duration_s = max(0.0, float(finished_at) - float(started_at))
        remaining_after_s = deadline.remaining_s()
        if remaining_before_s is None:
            remaining_before_s = remaining_after_s + duration_s
        metric: dict[str, object] = {
            "stage": stage,
            "attempt": max(0, int(attempt)),
            "started_at": float(started_at),
            "finished_at": float(finished_at),
            "duration_ms": duration_s * 1000.0,
            "status": status,
            "deadline_remaining_before_ms": max(0.0, remaining_before_s) * 1000.0,
            "deadline_remaining_after_ms": max(0.0, remaining_after_s) * 1000.0,
        }
        for key, value in (
            ("plan_id", plan_id),
            ("lane", lane),
            ("query", query),
            ("provider", provider),
            ("evidence_id", evidence_id),
            ("candidate_id", candidate_id),
        ):
            if value is not None:
                metric[key] = value
        ctx.phase_metrics.append(metric)

    @staticmethod
    def _runtime_diagnostics(
        *,
        request: RunRequest,
        deadline: _RunDeadline,
        ctx: _RunContext | None,
        terminal_at: float,
        deadline_met: bool,
    ) -> dict[str, object]:
        retrieval = ctx.retrieval_budget if ctx is not None else retrieval_budget_for(request.mode)
        deadline_overrun_ms = max(0.0, (terminal_at - float(deadline.deadline_at)) * 1000.0)
        summary = {
            "requested_at": float(request.requested_at),
            "deadline_at": float(deadline.deadline_at),
            "worker_started_at": float(deadline.worker_started_at),
            "writer_started_at": ctx.writer_started_at if ctx is not None else None,
            "terminal_at": float(terminal_at),
            "queue_delay_ms": max(
                0.0,
                (float(deadline.worker_started_at) - float(request.requested_at)) * 1000.0,
            ),
            "writer_reserve_s": float(retrieval.writer_reserve_s),
            "finalize_reserve_s": float(retrieval.finalize_reserve_s),
            "cleanup_reserve_s": float(retrieval.cleanup_reserve_s),
            "deadline_met": bool(deadline_met and deadline_overrun_ms <= 0),
            "cleanup_overrun_ms": deadline_overrun_ms,
        }
        retrieval_payload = retrieval.to_dict()
        if ctx is not None and ctx.retrieval_rejection_reason_codes:
            retrieval_payload["rejection_reason_codes"] = sorted(
                ctx.retrieval_rejection_reason_codes
            )
        writer_diagnostics = dict(ctx.writer_diagnostics) if ctx is not None else {}
        if writer_diagnostics:
            writer_diagnostics.setdefault("writer_started_at", ctx.writer_started_at)
            writer_diagnostics.setdefault("deadline_at", float(deadline.deadline_at))
        return {
            "phase_metrics": list(ctx.phase_metrics) if ctx is not None else [],
            "deadline_summary": summary,
            "retrieval_budget": retrieval_payload,
            "retrieval_outcomes": (
                [item.to_dict() for item in ctx.retrieval_outcomes]
                if ctx is not None
                else []
            ),
            "writer_diagnostics": writer_diagnostics,
            "kg_retrieval": (
                ctx.kg_retrieval.to_dict()
                if ctx is not None
                else Orchestrator._initial_kg_retrieval_diagnostics(
                    request
                ).to_dict()
            ),
        }

    def _complete_grounded_partial_timeout(
        self,
        *,
        request: RunRequest,
        rid: str,
        sink: EventSink | None,
        ctx: _RunContext,
        deadline: _RunDeadline,
    ) -> RunOutcome:
        """Commit an already-validated partial without starting a post-deadline writer."""

        answer = ctx.grounded_partial_answer
        if answer is None:  # defensive; caller checks the invariant
            self._raise_timeout()
        if ctx.answer_stream is not None and ctx.answer_stream.draft_active:
            ctx.answer_stream.reset("deadline_exceeded")
        extra: dict[str, object] = {
            **ctx.grounded_partial_extra,
            "completion_reason": "timeout",
            "execution_provenance": self._execution_provenance(ctx),
            "retrieval_outcomes": [item.to_dict() for item in ctx.retrieval_outcomes],
            "degraded_reasons": list(dict.fromkeys(ctx.degraded_reasons)),
            "research_intent": ctx.research_intent.to_dict() if ctx.research_intent else None,
            "evidence_eligibility": [item.to_dict() for item in ctx.evidence_eligibility.values()],
            "remediation": ctx.remediation.to_dict(),
            "recovery_searches_max": ctx.recovery_searches_max,
            "recovery_searches_used": ctx.recovery_searches_used,
            "resolved_search_usage": dict(request.resolved_search_usage),
        }
        extra["verification"] = {"status": "running", "started_at": time.time()}
        answer.confidence_label = ""
        self._start_verification(request, ctx.grounded_partial_evidence, ctx)
        ctx.kg_retrieval.mark_adopted(ctx.grounded_partial_evidence)
        extra["kg_retrieval"] = ctx.kg_retrieval.to_dict()
        stream_metrics = self._stream_metrics(ctx, request)
        extra["stream_metrics"] = stream_metrics
        diagnostics = self._runtime_diagnostics(
            request=request,
            deadline=deadline,
            ctx=ctx,
            terminal_at=time.time(),
            deadline_met=False,
        )
        extra.update(diagnostics)
        outcome = RunOutcome(
            run_id=rid,
            mode=request.mode.value,
            answer=answer,
            status="partial_timeout",
            extra=extra,
        )
        if not self._store.transition_run(
            rid,
            expected_statuses={"running"},
            status="partial_timeout",
            result=outcome.to_dict(),
            metadata_patch={"stream_metrics": stream_metrics, **diagnostics, **self._t48_audit_metadata(ctx)},
            citation_rows=self._citation_rows(answer.citations),
        ):
            raise BaselineError(
                ErrorCode.RUN_CANCELLED,
                "run reached a terminal state before partial finalization completed",
            )
        self._promote_final_external_evidence(ctx.grounded_partial_evidence)
        if ctx.answer_stream is not None:
            ctx.answer_stream.commit()
        self._emit(
            rid,
            "workflow_end",
            {"status": "partial_timeout", "code": ErrorCode.RUN_TIMEOUT.value},
            sink,
        )
        self._complete_verification(request, outcome, ctx)
        return outcome

    def _stream_cancel_checker(
        self,
        run_id: str,
        deadline: _RunDeadline | None = None,
    ) -> Callable[[], None]:
        """Return a chunk-boundary cancel probe with a 100ms persisted-status cache."""

        last_persisted_check = 0.0
        persisted_cancelled = False
        cache_lock = threading.Lock()

        def _check() -> None:
            nonlocal last_persisted_check, persisted_cancelled
            with cache_lock:
                with self._lock:
                    local_cancelled = run_id in self._cancelled
                if local_cancelled:
                    raise BaselineError(ErrorCode.RUN_CANCELLED, "run was cancelled")
                now = time.monotonic()
                if now - last_persisted_check >= 0.1:
                    run = self._store.get_run(run_id)
                    persisted_cancelled = bool(run and run.get("status") == "cancelling")
                    last_persisted_check = now
                if persisted_cancelled:
                    raise BaselineError(ErrorCode.RUN_CANCELLED, "run was cancelled")
                if deadline is not None and deadline.expired():
                    self._raise_timeout()

        return _check

    def _network_interrupt_checker(
        self,
        run_id: str,
        deadline: _RunDeadline | None,
    ) -> Callable[[], None]:
        """Return a Store-free probe safe for pure-network worker threads.

        Persisted cancellation is polled by the orchestrator thread while it waits
        for the future.  Network workers may only observe the in-process flag and
        monotonic deadline so they never share the SQLite connection with local or
        memory retrieval.
        """

        def _check() -> None:
            with self._lock:
                if run_id in self._cancelled:
                    raise BaselineError(ErrorCode.RUN_CANCELLED, "run was cancelled")
            if deadline is not None and deadline.expired():
                self._raise_timeout()

        return _check

    def _await_network_future(
        self,
        future: Future,
        *,
        run_id: str,
        deadline: _RunDeadline | None,
        reserve_s: float = 0.0,
    ):
        """Wait in bounded slices so cross-process cancellation stays observable."""

        while True:
            self._check_cancel(run_id)
            if deadline is not None:
                remaining = deadline.remaining_s(reserve_s=reserve_s)
                if remaining <= 0:
                    self._raise_timeout()
                wait_s = min(0.1, remaining)
            else:
                wait_s = 0.1
            try:
                return future.result(timeout=wait_s)
            except FutureTimeoutError:
                # On modern Python concurrent.futures.TimeoutError aliases the
                # built-in TimeoutError.  If the future is already done, re-read it
                # without a wait so an upstream TimeoutError is not mistaken for our
                # polling interval.
                if future.done():
                    return future.result()

    # --- per-run resolution ---------------------------------------------------

    def _build_context(self, request: RunRequest) -> _RunContext:
        """Resolve the LLM, composer, and council members for a single run."""

        llm = self._llm
        if self._llm_resolver is not None:
            resolved = self._llm_resolver(request)
            if resolved is not None:
                llm = resolved
        # Only resolve council members for council runs: the chair may invoke the
        # LLM to assign perspectives, which would be wasted work for other modes.
        members: list = []
        if request.mode is Mode.COUNCIL and self._council_resolver is not None:
            members = self._council_resolver(request) or []
        if not members:
            members = self._council_members or [llm]
        params = self._params_resolver(request) if self._params_resolver is not None else LLMCallParams()
        adapters = [llm]
        adapters.extend(member.adapter if isinstance(member, CouncilMemberSpec) else member for member in members)
        refs = tuple(dict.fromkeys(ref for adapter in adapters if (ref := _adapter_model_ref(adapter))))
        return _RunContext(
            llm=llm,
            composer=AnswerComposer(llm),
            members=members,
            llm_params=params,
            conversation_context=self._continuity_context(request),
            llm_model_refs=refs,
        )

    def _resolve_context_budget(self, request: RunRequest) -> ContextBudgetConfig:
        """Resolve the effective budget for a run.

        Prefers the run's already-resolved override (set by the API in
        ``create_run``); falls back to the orchestrator's instance config, then the
        deploy-level ``DEEPRESEARCH_CONTEXT_*`` env, then the built-in default. The
        per-run override is trusted (already validated); a malformed value degrades
        to the instance/env/default chain rather than failing the run.
        """

        override = getattr(request, "context_budget", None)
        if override:
            try:
                return coerce_payload(override)
            except ValueError:
                pass
        if self._context_budget is not None:
            return self._context_budget
        return from_env(os.environ) or default_config()

    def _continuity_context(self, request: RunRequest) -> str:
        """Assemble the prompt continuity context across the shared char budget.

        Priority order, each block bounded by its own cap and the remaining total:
        recent linear turns + semantic same-session recall (DR-ITER-053), then
        long-term workspace/user memory (DR-ITER-054). Returns "" when nothing
        applies, leaving single-turn / no-policy runs byte-for-byte unchanged.
        """

        cfg = self._resolve_context_budget(request)
        parts: list[str] = []
        used = 0
        conversation = self._conversation_context(request, cfg, budget=cfg.max_chars)
        if conversation:
            parts.append(conversation)
            used += len(conversation)
        memory = self._memory_context(
            request, cfg, budget=min(cfg.memory_chars, max(0, cfg.max_chars - used))
        )
        if memory:
            parts.append(memory)
            used += len(memory)
        library = self._library_context(request, cfg, budget=max(0, cfg.max_chars - used))
        if library:
            parts.append(library)
        return "\n\n".join(parts)

    def _library_context(self, request: RunRequest, cfg: ContextBudgetConfig, *, budget: int) -> str:
        """Recall the optional non-citable Library background block.

        The independent local evidence lane remains active regardless of this
        context setting and may cite eligible note/report artifacts.
        """

        if budget <= 0 or self._library_retriever is None:
            return ""
        scope = getattr(request, "library_scope", "off")
        if scope == "off":
            return ""
        try:
            hits = self._library_retriever.retrieve(request.query, scope=scope, top_k=request.resolved_budget().top_k)
        except Exception:  # noqa: BLE001 - Library context is best-effort
            return ""
        cap = min(budget, cfg.memory_chars, 3000)
        return render_library_context(hits, max_chars=cap)

    def _memory_context(self, request: RunRequest, cfg: ContextBudgetConfig, *, budget: int) -> str:
        """Recall long-term memory for the run as a labeled, non-citable block.

        Honors ``request.memory_policy`` (``off`` injects nothing; ``workspace``
        pulls the workspace layer only; ``workspace_user`` adds the user layer).
        Best-effort: any failure or empty recall yields "". This background path
        never writes into the evidence pack and is explicitly marked do-not-cite;
        the independent Memory evidence lane may still cite active/archived items.
        """

        if budget <= 0:
            return ""
        policy = getattr(request, "memory_policy", MEMORY_POLICY_DEFAULT)
        if policy == "off" or self._memory is None:
            return ""
        user_key = request.user_key if policy == "workspace_user" else ""
        try:
            memories = self._memory.recall(
                session_key="",
                workspace_key=request.workspace_key,
                user_key=user_key,
                limit=8,
                query=request.query,
            )
            if user_key:
                # The synthesized user profile is policy-selected global context,
                # not a query-recall candidate. Older profile rows have no embedding;
                # keep that established workspace_user contract without bypassing the
                # relevance floor for ordinary unembedded memories.
                recalled_ids = {str(memory.get("id") or "") for memory in memories}
                profile = next(
                    (
                        row
                        for row in self._store.list_memories(
                            namespace="user", scope_key=user_key, limit=20
                        )
                        if row.get("metadata", {}).get("source") == "user_profile_v1"
                        and str(row.get("id") or "") not in recalled_ids
                    ),
                    None,
                )
                if profile is not None:
                    memories.append(profile)
        except Exception:  # noqa: BLE001 - memory recall is best-effort, never fatal
            return ""
        lines: list[str] = []
        total = 0
        for mem in memories:
            content = (mem.get("content") or "").strip()
            if not content:
                continue
            content = " ".join(content.split())
            if len(content) > _CONTEXT_MSG_CHARS:
                content = content[:_CONTEXT_MSG_CHARS].rstrip() + "…"
            line = f"- {content}"
            if lines and total + len(line) > budget:
                break
            lines.append(line)
            total += len(line)
        if not lines:
            return ""
        return "Relevant memories (do not cite as sources):\n" + "\n".join(lines)

    def _conversation_context(
        self, request: RunRequest, cfg: ContextBudgetConfig, *, budget: int | None = None
    ) -> str:
        """Build the session's continuity transcript for follow-ups.

        Two blocks share ``budget`` (priority: recent -> semantic): the most recent
        linear turns (``_format_history``) and, when an embedder is available, the
        most semantically relevant *earlier same-session* turns recalled via
        ``ConversationIndex.search_history`` (intra scope). Semantic hits are deduped
        against the recent block by message id so the latest round is never injected
        twice. The current run's own messages are persisted *after* completion, so
        this sees strictly prior turns. Degrades to an empty string when there is no
        session history; without an embedder/semantic hits the output is identical to
        the linear-only behavior (DR-ITER-026A), keeping single-turn runs unchanged.
        """

        if not request.session_id:
            return ""
        if budget is None:
            budget = cfg.max_chars
        try:
            messages = self._store.list_messages(request.session_id)
        except Exception:  # noqa: BLE001 - context is best-effort, never fatal
            return ""
        recent_cap = min(cfg.recent_chars, max(0, budget))
        recent = _format_history(
            messages, turns=cfg.turns, max_chars=recent_cap, msg_chars=_CONTEXT_MSG_CHARS
        )
        blocks: list[str] = []
        used = 0
        if recent:
            blocks.append(recent)
            used += len(recent)
        semantic = self._semantic_history_block(
            request, messages, cfg, budget=min(cfg.semantic_chars, max(0, budget - used))
        )
        if semantic:
            blocks.append(semantic)
        return "\n\n".join(blocks)

    def _semantic_history_block(
        self, request: RunRequest, messages: list[dict], cfg: ContextBudgetConfig, *, budget: int
    ) -> str:
        """Recall earlier same-session turns relevant to the query, as a labeled block.

        Returns ``""`` when there is no query, no budget, no embedder, or no hit that
        falls outside the recent linear window (deduped by message id).
        """

        if budget <= 0 or not request.query.strip():
            return ""
        recent_ids = {
            str(m.get("id"))
            for m in messages[-cfg.turns:]
            if m.get("role") in ("user", "assistant")
        }
        # A new session, or one whose complete transcript already fits in the
        # recent window, has no earlier turn that semantic recall could add.
        # Skipping the embed call preserves the same prompt while protecting the
        # request-level Live deadline from provider latency with zero UX benefit.
        has_older_candidate = any(
            m.get("role") in ("user", "assistant")
            and str(m.get("content") or "").strip()
            and str(m.get("id")) not in recent_ids
            for m in messages
        )
        if not has_older_candidate:
            return ""
        try:
            hits = self._history.search_history(
                request.query, session_id=request.session_id, scope=SCOPE_INTRA, limit=5
            )
        except Exception:  # noqa: BLE001 - semantic recall is best-effort, never fatal
            return ""
        lines: list[str] = []
        total = 0
        for hit in hits:
            if str(hit.get("id")) in recent_ids:
                continue
            role = hit.get("role")
            content = (hit.get("content") or "").strip()
            if role not in ("user", "assistant") or not content:
                continue
            label = "User" if role == "user" else "Assistant"
            content = " ".join(content.split())
            if len(content) > _CONTEXT_MSG_CHARS:
                content = content[:_CONTEXT_MSG_CHARS].rstrip() + "…"
            line = f"{label}: {content}"
            if lines and total + len(line) > budget:
                break
            lines.append(line)
            total += len(line)
        if not lines:
            return ""
        return "Related earlier turns in this session (for continuity only; do not cite):\n" + "\n".join(lines)

    # --- dispatch -------------------------------------------------------------

    def _dispatch(
        self,
        request: RunRequest,
        budget: Budget,
        rid: str,
        sink: EventSink | None,
        ctx: _RunContext,
        deadline: _RunDeadline,
    ) -> RunOutcome:
        if request.mode is Mode.FAST:
            return self._run_fast(request, budget, rid, sink, ctx, deadline)
        if request.mode is Mode.MULTIPLE:
            return self._run_multiple(request, budget, rid, sink, ctx, deadline)
        if request.mode is Mode.DEEP:
            return self._run_deep(request, budget, rid, sink, ctx, deadline)
        if request.mode is Mode.COUNCIL:
            return self._run_council(request, budget, rid, sink, ctx, deadline)
        raise BaselineError(ErrorCode.INVALID_REQUEST, f"unsupported mode {request.mode}")

    def _run_fast(
        self,
        request: RunRequest,
        budget: Budget,
        rid: str,
        sink: EventSink | None,
        ctx: _RunContext,
        deadline: _RunDeadline,
    ) -> RunOutcome:
        intent = self._intent(ctx)
        self._check_deadline(deadline)
        self._emit(rid, "node_status", {"node": "search", "state": "running"}, sink)
        evidence = self._retrieve_evidence_plan(
            intent.research_focus,
            request,
            budget,
            rid,
            sink,
            ctx,
            plan_id="primary",
            deadline=deadline,
        )
        ctx.ordinary_searches_used = 1
        self._emit_budget(
            rid,
            sink,
            mode=request.mode,
            used=1,
            maximum=budget.max_searches,
            stage="search",
            recovery_searches_max=1,
            recovery_searches_used=ctx.recovery_searches_used,
        )
        self._check_cancel(rid)
        self._emit(rid, "node_status", {"node": "retrieve", "state": "running"}, sink)
        evidence = self._prepare_adopted_evidence(
            evidence,
            intent=intent,
            request=request,
            budget=budget,
            rid=rid,
            sink=sink,
            ctx=ctx,
            deadline=deadline,
            stage="primary",
        )
        evidence, temporal = self._temporal_assurance(
            request,
            budget,
            rid,
            sink,
            ctx,
            evidence,
            plan_id="temporal",
            deadline=deadline,
        )
        timed_out = deadline.expired()
        if timed_out and not evidence:
            self._raise_timeout()
        evidence = self._prepare_adopted_evidence(
            evidence,
            intent=intent,
            request=request,
            budget=budget,
            rid=rid,
            sink=sink,
            ctx=ctx,
            deadline=deadline,
            stage="post_temporal",
        )
        evidence, temporal, conflict_analysis = self._final_evidence_snapshot(
            request,
            evidence,
            rid=rid,
            sink=sink,
            ctx=ctx,
            deadline=deadline,
        )
        self._emit(rid, "node_status", {"node": "compose", "state": "running"}, sink)
        answer, evidence, temporal, conflict_analysis = self._compose_with_repair(
            request=request,
            budget=budget,
            rid=rid,
            sink=sink,
            ctx=ctx,
            deadline=deadline,
            evidence=evidence,
            temporal=temporal,
            conflict_analysis=conflict_analysis,
            structured="brief" if intent.output_shape == "default" else False,
            context=ctx.conversation_context,
        )
        extra = {
            "sources": _sources(evidence),
            "retrieval_outcomes": [item.to_dict() for item in ctx.retrieval_outcomes],
            "degraded_reasons": list(dict.fromkeys(ctx.degraded_reasons)),
            "temporal_coverage": temporal.to_dict(),
        }
        if timed_out:
            extra["completion_reason"] = "timeout"
        return RunOutcome(
            run_id=rid, mode=request.mode.value, answer=answer,
            status=terminal_status(timed_out=timed_out, degraded_reasons=ctx.degraded_reasons),
            extra=extra,
        )

    def _run_multiple(
        self,
        request: RunRequest,
        budget: Budget,
        rid: str,
        sink: EventSink | None,
        ctx: _RunContext,
        deadline: _RunDeadline,
    ) -> RunOutcome:
        intent = self._intent(ctx)
        retrieval = ctx.retrieval_budget or retrieval_budget_for(request.mode)
        self._check_deadline(deadline, reserve_s=retrieval.reserve_total_s)
        with use_provider_call_budget(
            deadline.provider_budget(
                phase_reserve_s=retrieval.reserve_total_s,
                retrieval_budget=retrieval,
                interrupt_check=self._stream_cancel_checker(rid, deadline),
            )
        ):
            subqueries = self._subqueries(
                intent.research_focus,
                budget.max_searches,
                ctx,
                options=request.search_options,
            )
        subqueries = _source_constrained_subqueries(
            intent,
            subqueries,
            count=budget.max_searches,
        )
        all_evidence_rankings: list[list[EvidenceItem]] = []
        lanes: list[dict] = []
        timed_out = False
        prefetched: dict[int, _ExternalSearchHandle] = {}
        next_prefetch = 0
        prefetch_limit = max(1, min(int(budget.concurrency), len(subqueries)))
        search_network = getattr(self._search, "search_network", None)

        def prefetch_until_full() -> None:
            nonlocal next_prefetch
            if not callable(search_network):
                return
            while next_prefetch < len(subqueries) and len(prefetched) < prefetch_limit:
                self._check_cancel(rid)
                if deadline.expired(reserve_s=retrieval.reserve_total_s):
                    return
                handle = self._begin_external_search(
                    subqueries[next_prefetch],
                    request=request,
                    ctx=ctx,
                    rid=rid,
                    recovery=False,
                    excluded_provenance_roots=(),
                    relevance_query=None,
                    deadline=deadline,
                )
                if handle is None:
                    return
                prefetched[next_prefetch] = handle
                next_prefetch += 1
                ctx.ordinary_searches_used = max(ctx.ordinary_searches_used, next_prefetch)

        try:
            prefetch_until_full()
            for index, sub in enumerate(subqueries):
                self._check_cancel(rid)
                if deadline.expired(reserve_s=retrieval.reserve_total_s):
                    if not all_evidence_rankings:
                        self._raise_timeout()
                    ctx.retrieval_rejection_reason_codes.add("retrieval_stopped_for_writer_reserve")
                    break
                self._emit(rid, "node_status", {"node": f"lane_{index + 1}", "state": "running", "query": sub}, sink)
                search_handle = prefetched.pop(index, None)
                try:
                    lane_evidence = self._retrieve_evidence_plan(
                        sub,
                        request,
                        budget,
                        rid,
                        sink,
                        ctx,
                        plan_id=f"lane-{index + 1}",
                        deadline=deadline,
                        external_search_handle=search_handle,
                        external_search_started=search_handle is not None,
                    )
                finally:
                    self._close_external_search_handle(search_handle)
                if search_handle is None:
                    ctx.ordinary_searches_used = max(ctx.ordinary_searches_used, index + 1)
                prefetch_until_full()
                self._emit_budget(
                    rid, sink, mode=request.mode, used=ctx.ordinary_searches_used,
                    maximum=budget.max_searches, stage=f"lane_{index + 1}",
                    recovery_searches_max=1,
                    recovery_searches_used=ctx.recovery_searches_used,
                )
                all_evidence_rankings.append(lane_evidence)
                lanes.append({"query": sub, "sources": _sources(lane_evidence)})
                if deadline.expired(reserve_s=retrieval.reserve_total_s):
                    if not any(all_evidence_rankings):
                        self._raise_timeout()
                    ctx.retrieval_rejection_reason_codes.add("retrieval_stopped_for_writer_reserve")
                    break
        finally:
            for handle in prefetched.values():
                self._close_external_search_handle(handle)
        self._check_cancel(rid)
        self._emit(rid, "node_status", {"node": "merge", "state": "running"}, sink)
        merged = self._merge_evidence_rankings(all_evidence_rankings, budget.top_k)
        merged = self._prepare_adopted_evidence(
            merged,
            intent=intent,
            request=request,
            budget=budget,
            rid=rid,
            sink=sink,
            ctx=ctx,
            deadline=deadline,
            stage="multiple_merge",
        )
        merged, temporal = self._temporal_assurance(
            request,
            budget,
            rid,
            sink,
            ctx,
            merged,
            plan_id="temporal",
            deadline=deadline,
        )
        timed_out = timed_out or deadline.expired()
        if timed_out and not merged:
            self._raise_timeout()
        merged = self._prepare_adopted_evidence(
            merged,
            intent=intent,
            request=request,
            budget=budget,
            rid=rid,
            sink=sink,
            ctx=ctx,
            deadline=deadline,
            stage="multiple_post_temporal",
        )
        merged, temporal, conflict_analysis = self._final_evidence_snapshot(
            request,
            merged,
            rid=rid,
            sink=sink,
            ctx=ctx,
            deadline=deadline,
        )
        self._emit(rid, "node_status", {"node": "compose", "state": "running"}, sink)
        answer, merged, temporal, conflict_analysis = self._compose_with_repair(
            request=request,
            budget=budget,
            rid=rid,
            sink=sink,
            ctx=ctx,
            deadline=deadline,
            evidence=merged,
            temporal=temporal,
            conflict_analysis=conflict_analysis,
            structured=True if intent.output_shape == "default" else False,
            context=ctx.conversation_context,
        )
        extra = {
            "lanes": _filter_lanes_to_final_sources(lanes, merged),
            "sources": _sources(merged),
            "retrieval_outcomes": [item.to_dict() for item in ctx.retrieval_outcomes],
            "degraded_reasons": list(dict.fromkeys(ctx.degraded_reasons)),
            "temporal_coverage": temporal.to_dict(),
        }
        if timed_out:
            extra["completion_reason"] = "timeout"
        return RunOutcome(
            run_id=rid, mode=request.mode.value, answer=answer,
            status=terminal_status(timed_out=timed_out, degraded_reasons=ctx.degraded_reasons),
            extra=extra,
        )

    def _deep_checkpoint_boundary(
        self,
        stage: str,
        *,
        request: RunRequest,
        rid: str,
        sink: EventSink | None,
        ctx: _RunContext,
        state: ResearchState,
        section_outputs: list[dict],
        collected: list[EvidenceItem],
        collected_rankings: list[list[EvidenceItem]],
        gap_tasks: set[str],
        temporal: TemporalCoverage | None = None,
        conflict_analysis: tuple[list, list[dict]] | None = None,
    ) -> None:
        """Persist one complete Deep safe-boundary snapshot and honor user Pause."""

        self._check_cancel(rid)
        run = self._store.get_run(rid)
        status = str((run or {}).get("status") or "")
        user_pause = status == "pausing"
        if not user_pause and not (
            status == "running" and request.auto_checkpoint_enabled
        ):
            return
        previous_events = self._store.list_events(rid)
        after_seq = int(previous_events[-1]["seq"]) if previous_events else 0
        try:
            checkpoint = self._build_deep_checkpoint(
                stage,
                request=request,
                rid=rid,
                ctx=ctx,
                state=state,
                section_outputs=section_outputs,
                collected=collected,
                collected_rankings=collected_rankings,
                gap_tasks=gap_tasks,
                temporal=temporal,
                conflict_analysis=conflict_analysis,
            )
            encoded = encode_deep_checkpoint(checkpoint)
            if user_pause:
                saved = self._store.save_deep_checkpoint_and_pause(
                    run_id=rid,
                    lineage_id=checkpoint.lineage_id,
                    stage=stage,
                    schema_version=CHECKPOINT_SCHEMA_VERSION,
                    payload_json=encoded.payload_json,
                    payload_sha256=encoded.payload_sha256,
                )
            else:
                saved = self._store.save_deep_checkpoint(
                    run_id=rid,
                    lineage_id=checkpoint.lineage_id,
                    stage=stage,
                    schema_version=CHECKPOINT_SCHEMA_VERSION,
                    payload_json=encoded.payload_json,
                    payload_sha256=encoded.payload_sha256,
                )
                raced = self._store.get_run(rid)
                if str((raced or {}).get("status") or "") == "pausing":
                    saved = self._store.save_deep_checkpoint_and_pause(
                        run_id=rid,
                        lineage_id=checkpoint.lineage_id,
                        stage=stage,
                        schema_version=CHECKPOINT_SCHEMA_VERSION,
                        payload_json=encoded.payload_json,
                        payload_sha256=encoded.payload_sha256,
                    )
                    user_pause = saved is not None
        except Exception:
            if user_pause:
                self._store.rollback_run_pause(
                    rid,
                    reason="checkpoint_write_failed",
                )
            else:
                self._emit(
                    rid,
                    "checkpoint_save_failed",
                    {"reason": "automatic_checkpoint_failed", "stage": stage},
                    sink,
                )
                return
            self._forward_persisted_events(rid, after_seq=after_seq, sink=sink)
            return
        self._forward_persisted_events(rid, after_seq=after_seq, sink=sink)
        if user_pause and saved is not None:
            raise DeepRunPaused()
        self._check_cancel(rid)

    def _build_deep_checkpoint(
        self,
        stage: str,
        *,
        request: RunRequest,
        rid: str,
        ctx: _RunContext,
        state: ResearchState,
        section_outputs: list[dict],
        collected: list[EvidenceItem],
        collected_rankings: list[list[EvidenceItem]],
        gap_tasks: set[str],
        temporal: TemporalCoverage | None,
        conflict_analysis: tuple[list, list[dict]] | None,
    ) -> DeepCheckpointV1:
        conflicts, conflict_decisions = conflict_analysis or ([], [])
        evidence = _dedup_evidence(collected)
        if stage == "final_evidence_ready":
            ctx.final_adopted_evidence = list(evidence)
        return DeepCheckpointV1(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            stage=stage,
            lineage_id=request.lineage_id or rid,
            parent_run_id=rid,
            request=request.to_payload(),
            research_intent=self._intent(ctx).to_dict(),
            plan=state.to_checkpoint(),
            pending_tasks=[{"title": title} for title in state._pending],
            completed_sections=[dict(item) for item in section_outputs],
            evidence_snapshot=[item.to_dict() for item in evidence],
            retrieval_state={
                "research_state": state.to_checkpoint(),
                "collected_rankings": [
                    [item.to_dict() for item in ranking]
                    for ranking in collected_rankings
                ],
                "gap_tasks": sorted(gap_tasks),
                "retrieval_outcomes": [
                    item.to_dict() for item in ctx.retrieval_outcomes
                ],
                "kg_retrieval": ctx.kg_retrieval.to_checkpoint_dict(),
            },
            temporal_coverage=temporal.to_dict() if temporal is not None else {},
            conflicts=[
                item.to_dict() if hasattr(item, "to_dict") else dict(item)
                for item in conflicts
            ],
            evidence_eligibility=[
                item.to_dict() for item in ctx.evidence_eligibility.values()
            ],
            grounding={
                "final_evidence_ids": [evidence_identity(item) for item in evidence],
                "degraded_reasons": list(dict.fromkeys(ctx.degraded_reasons)),
                "retrieval_rejection_reason_codes": sorted(
                    ctx.retrieval_rejection_reason_codes
                ),
                "conflict_decisions": [
                    _checkpoint_conflict_decision(item)
                    for item in conflict_decisions
                    if isinstance(item, dict)
                ],
            },
            remediation=ctx.remediation.to_dict(),
            phase_audit=[
                _checkpoint_phase_metric(item)
                for item in ctx.phase_metrics
                if isinstance(item, dict)
            ],
            budget_usage={
                "searches_used": state.searches_used,
                "searches_max": state.max_searches,
                "pages_fetched": state.pages_fetched,
                "think_tokens_used": state.think_tokens_used,
                "gap_rounds": state.gap_rounds,
                "ordinary_searches_used": ctx.ordinary_searches_used,
                "recovery_searches_used": ctx.recovery_searches_used,
                "recovery_searches_max": ctx.recovery_searches_max,
                "providers_attempted": list(ctx.providers_attempted),
            },
        )

    def _restore_deep_checkpoint(
        self,
        request: RunRequest,
        *,
        rid: str,
        ctx: _RunContext,
    ) -> tuple[
        str,
        ResearchState,
        list[dict],
        list[EvidenceItem],
        list[list[EvidenceItem]],
        set[str],
    ] | None:
        """Restore the claimed parent checkpoint without carrying its old deadline."""

        parent_run_id = request.checkpoint_parent_run_id
        if not parent_run_id:
            return None
        checkpoint = self._store.get_deep_checkpoint(
            parent_run_id,
            include_payload=True,
        )
        if (
            checkpoint is None
            or checkpoint.get("integrity") != "valid"
            or int(checkpoint.get("revision") or 0) != request.checkpoint_revision
            or str(checkpoint.get("disposition") or "") != "claimed"
            or str(checkpoint.get("claimed_child_run_id") or "") != rid
        ):
            raise BaselineError(
                ErrorCode.CHECKPOINT_STATE_CONFLICT,
                "claimed Deep checkpoint is no longer resumable",
            )
        payload = checkpoint.get("payload")
        if not isinstance(payload, dict):
            raise BaselineError(
                ErrorCode.CHECKPOINT_INVALID,
                "Deep checkpoint payload is unavailable",
            )
        if str(payload.get("lineage_id") or "") != request.lineage_id:
            raise BaselineError(
                ErrorCode.CHECKPOINT_STATE_CONFLICT,
                "Deep checkpoint lineage changed before resume",
            )
        retrieval_state = payload.get("retrieval_state")
        if not isinstance(retrieval_state, dict):
            raise BaselineError(
                ErrorCode.CHECKPOINT_INVALID,
                "Deep checkpoint retrieval state is unavailable",
            )
        try:
            state = ResearchState.from_checkpoint(
                retrieval_state.get("research_state") or payload.get("plan")
            )
        except (TypeError, ValueError) as error:
            raise BaselineError(
                ErrorCode.CHECKPOINT_INVALID,
                "Deep checkpoint research state is invalid",
            ) from error
        incremental = request.checkpoint_incremental_budget or {}
        if "max_searches" in incremental:
            try:
                added_searches = max(0, int(incremental["max_searches"]))
            except (TypeError, ValueError) as error:
                raise BaselineError(
                    ErrorCode.INVALID_REQUEST,
                    "incremental max_searches must be an integer",
                ) from error
            state.max_searches = state.searches_used + added_searches
        if request.checkpoint_action == "revise":
            remaining = request.checkpoint_remaining_plan or {}
            revised = [
                str(item).strip()
                for item in (remaining.get("sections") or [])
                if str(item).strip()
            ]
            completed = [
                item
                for item in state.plan
                if item not in set(state._pending)
            ]
            state.plan = list(dict.fromkeys([*completed, *revised]))
            state._pending = list(dict.fromkeys(revised))
            state.max_searches = max(
                state.searches_used,
                state.max_searches,
                state.searches_used + len(state._pending),
            )
        section_outputs = [
            dict(item)
            for item in (payload.get("completed_sections") or [])
            if isinstance(item, dict)
        ]
        collected = self._restore_checkpoint_evidence(
            payload.get("evidence_snapshot"),
            ctx=ctx,
        )
        rankings: list[list[EvidenceItem]] = []
        for raw_ranking in retrieval_state.get("collected_rankings") or []:
            restored_ranking = self._restore_checkpoint_evidence(
                raw_ranking,
                ctx=ctx,
            )
            if restored_ranking:
                rankings.append(restored_ranking)
        if not rankings and collected:
            rankings = [list(collected)]
        outcomes: list[RetrievalOutcome] = []
        for item in retrieval_state.get("retrieval_outcomes") or []:
            if not isinstance(item, dict):
                continue
            try:
                outcomes.append(_retrieval_outcome_from_checkpoint(item))
            except (TypeError, ValueError):
                continue
        ctx.retrieval_outcomes = outcomes
        restored_kg = KGRetrievalDiagnostics.from_checkpoint_dict(
            retrieval_state.get("kg_retrieval"),
            requested=request.kg_aware_retrieval_enabled,
        )
        ctx.kg_retrieval.merge(restored_kg)
        grounding = payload.get("grounding")
        if isinstance(grounding, dict):
            ctx.degraded_reasons = [
                str(item) for item in (grounding.get("degraded_reasons") or [])
            ] + ctx.degraded_reasons
            ctx.retrieval_rejection_reason_codes.update(
                str(item)
                for item in (
                    grounding.get("retrieval_rejection_reason_codes") or []
                )
            )
        budget_usage = payload.get("budget_usage")
        if isinstance(budget_usage, dict):
            ctx.ordinary_searches_used = max(
                state.searches_used,
                int(budget_usage.get("ordinary_searches_used") or 0),
            )
            ctx.recovery_searches_used = min(
                ctx.recovery_searches_max,
                max(0, int(budget_usage.get("recovery_searches_used") or 0)),
            )
            ctx.providers_attempted = [
                str(item)
                for item in (budget_usage.get("providers_attempted") or [])
            ]
        remediation = payload.get("remediation")
        if isinstance(remediation, dict):
            ctx.remediation = _remediation_from_checkpoint(remediation)
        ctx.phase_metrics = [
            dict(item)
            for item in (payload.get("phase_audit") or [])
            if isinstance(item, dict)
        ]
        return (
            str(payload.get("stage") or ""),
            state,
            section_outputs,
            collected,
            rankings,
            {
                str(item)
                for item in (retrieval_state.get("gap_tasks") or [])
                if str(item)
            },
        )

    def _restore_checkpoint_evidence(
        self,
        raw_items: object,
        *,
        ctx: _RunContext,
    ) -> list[EvidenceItem]:
        restored: list[EvidenceItem] = []
        if not isinstance(raw_items, list):
            return restored
        for value in raw_items:
            if not isinstance(value, dict):
                continue
            try:
                item = _evidence_from_checkpoint(value)
            except (TypeError, ValueError):
                continue
            if item.artifact_type == "memory":
                if not item.artifact_id or self._store.get_memory(item.artifact_id) is None:
                    ctx.degraded_reasons.append("checkpoint_memory_missing")
                    continue
            elif item.artifact_type == "library_document":
                if not item.artifact_id or self._store.get_document(item.artifact_id) is None:
                    ctx.degraded_reasons.append("checkpoint_source_tombstoned")
                    continue
            restored.append(item)
        return _dedup_evidence(restored)

    def _forward_persisted_events(
        self,
        run_id: str,
        *,
        after_seq: int,
        sink: EventSink | None,
    ) -> None:
        if sink is None:
            return
        for event in self._store.list_events(run_id, after_seq=after_seq):
            sink(
                {
                    "seq": event["seq"],
                    "type": event["type"],
                    "payload": event["payload"],
                }
            )

    def _run_deep(
        self,
        request: RunRequest,
        budget: Budget,
        rid: str,
        sink: EventSink | None,
        ctx: _RunContext,
        deadline: _RunDeadline,
    ) -> RunOutcome:
        intent = self._intent(ctx)
        retrieval = ctx.retrieval_budget or retrieval_budget_for(request.mode)
        restored = self._restore_deep_checkpoint(request, rid=rid, ctx=ctx)
        max_rounds = _deep_section_round_limit(intent, budget)
        explicit_section_count = (
            max_rounds
            if intent.output_shape == "sentence_count" and intent.sentence_count
            else None
        )
        if restored is None:
            self._check_deadline(deadline, reserve_s=retrieval.reserve_total_s)
            self._emit(rid, "node_status", {"node": "planner", "state": "running"}, sink)
            with use_provider_call_budget(
                deadline.provider_budget(
                    phase_reserve_s=retrieval.reserve_total_s,
                    retrieval_budget=retrieval,
                    interrupt_check=self._stream_cancel_checker(rid, deadline),
                )
            ):
                sections = self._plan(
                    intent.research_focus,
                    budget,
                    ctx,
                    section_count=explicit_section_count,
                )
            if explicit_section_count is not None and max_rounds < budget.max_searches:
                ctx.retrieval_rejection_reason_codes.add("deep_sections_capped_for_output_shape")
            section_outputs: list[dict] = []
            collected: list[EvidenceItem] = []
            collected_rankings: list[list[EvidenceItem]] = []
            gap_tasks: set[str] = set()
            state = ResearchState.from_plan(sections, max_searches=max_rounds)
            resume_stage = "plan_ready"
            self._deep_checkpoint_boundary(
                "plan_ready",
                request=request,
                rid=rid,
                sink=sink,
                ctx=ctx,
                state=state,
                section_outputs=section_outputs,
                collected=collected,
                collected_rankings=collected_rankings,
                gap_tasks=gap_tasks,
            )
        else:
            (
                resume_stage,
                state,
                section_outputs,
                collected,
                collected_rankings,
                gap_tasks,
            ) = restored
            max_rounds = state.max_searches
        section_tokens = max(400, budget.max_output_tokens // max(1, max_rounds))
        timed_out = False

        while resume_stage != "final_evidence_ready" and not state.done():
            title = state.next_subtask()
            if not title:
                break
            self._check_cancel(rid)
            section_start_reserve_s = retrieval.reserve_total_s + (
                _deep_section_work_reserve_s(deadline.timeout_s, retrieval)
                if collected
                else 0.0
            )
            if deadline.expired(reserve_s=section_start_reserve_s):
                if not collected:
                    self._raise_timeout()
                ctx.retrieval_rejection_reason_codes.add(
                    "deep_section_stopped_for_final_reserve"
                )
                break
            remaining = max_rounds - state.searches_used
            ctx.ordinary_searches_used = min(max_rounds, state.searches_used + 1)
            section_search_focus = _deep_section_search_focus(intent.research_focus, title)
            queries = expand_queries(section_search_focus, remaining_budget=remaining)
            section_rankings: list[list[EvidenceItem]] = []
            stopped_for_writer_reserve = False
            for query_index, query in enumerate(queries, start=1):
                if deadline.expired(reserve_s=retrieval.reserve_total_s):
                    ctx.retrieval_rejection_reason_codes.add(
                        "retrieval_stopped_for_writer_reserve"
                    )
                    stopped_for_writer_reserve = True
                    break
                self._emit(rid, "node_status", {"node": "planner", "state": "running", "title": title, "query": query}, sink)
                query_evidence = self._retrieve_evidence_plan(
                    query,
                    request,
                    budget,
                    rid,
                    sink,
                    ctx,
                    plan_id=f"deep-{state.searches_used + 1}-{query_index}",
                    relevance_query=section_search_focus,
                    deadline=deadline,
                )
                section_rankings.append(query_evidence)

            if stopped_for_writer_reserve and not section_rankings:
                if not collected:
                    self._raise_timeout()
                break

            evidence = _fuse_evidence_rankings(
                section_rankings,
                budget.top_k,
                preserve_ranking_diversity=True,
                merge_complementary_text=True,
            )
            analysis_title = title
            original_assessment = self._assess_and_record(
                intent, evidence, rid, sink, ctx, stage=f"deep_section_focus:{title}",
            )
            evidence = list(original_assessment.adopted)
            section_intent = replace(
                intent,
                research_focus=_deep_section_focus(
                    intent.research_focus,
                    analysis_title,
                ),
                citation_limit=None,
                output_shape="default",
                sentence_count=None,
                max_length=None,
            )
            evidence = self._prepare_adopted_evidence(
                evidence,
                intent=section_intent,
                request=request,
                budget=budget,
                rid=rid,
                sink=sink,
                ctx=ctx,
                deadline=deadline,
                stage=f"deep_section:{analysis_title}",
            )
            if not evidence:
                state.add_evidence(
                    title,
                    queries=queries,
                    notes="",
                    source_count=0,
                    status="evidence_ineligible",
                    think_tokens=0,
                )
                self._emit_budget(
                    rid,
                    sink,
                    mode=request.mode,
                    used=state.searches_used,
                    maximum=max_rounds,
                    stage="evidence_ineligible",
                    gap_rounds=state.gap_rounds,
                    pages_fetched=state.pages_fetched,
                    remaining_searches=max(0, max_rounds - state.searches_used),
                    recovery_searches_max=1,
                    recovery_searches_used=ctx.recovery_searches_used,
                )
                self._deep_checkpoint_boundary(
                    "round_complete",
                    request=request,
                    rid=rid,
                    sink=sink,
                    ctx=ctx,
                    state=state,
                    section_outputs=section_outputs,
                    collected=collected,
                    collected_rankings=collected_rankings,
                    gap_tasks=gap_tasks,
                )
                continue
            collected.extend(evidence)
            collected_rankings.append(evidence)
            self._emit(
                rid,
                "node_status",
                {"node": "analyst", "state": "running", "title": analysis_title},
                sink,
            )
            self._check_deadline(deadline, reserve_s=retrieval.reserve_total_s)
            with use_provider_call_budget(
                deadline.provider_budget(
                    phase_reserve_s=retrieval.reserve_total_s,
                    retrieval_budget=retrieval,
                    interrupt_check=self._stream_cancel_checker(rid, deadline),
                )
            ):
                notes = extract_notes(
                    ctx.llm,
                    analysis_title,
                    evidence,
                    max_tokens=section_tokens,
                    temperature=ctx.llm_params.temperature,
                    thinking_budget=ctx.llm_params.thinking_budget,
                )
            self._check_deadline(deadline, reserve_s=retrieval.reserve_total_s)
            with use_provider_call_budget(
                deadline.provider_budget(
                    phase_reserve_s=retrieval.reserve_total_s,
                    retrieval_budget=retrieval,
                    interrupt_check=self._stream_cancel_checker(rid, deadline),
                )
            ):
                part = ctx.composer.compose(
                    analysis_title, evidence, max_tokens=section_tokens,
                    instruction=f"Write the report section: {analysis_title}",
                    temperature=ctx.llm_params.temperature, thinking_budget=ctx.llm_params.thinking_budget,
                )
            self._check_deadline(deadline, reserve_s=retrieval.reserve_total_s)
            if part.text.strip():
                ctx.grounded_partial_answer = part
                ctx.grounded_partial_evidence = list(evidence)
                ctx.grounded_partial_extra = {
                    "sources": _sources(evidence),
                    "partial_stage": "deep_section",
                    "partial_section": analysis_title,
                }
            section_outputs.append(
                {
                    "title": analysis_title,
                    "answer": part.to_dict(),
                    "notes": notes,
                    "_evidence_keys": [
                        key
                        for item in evidence
                        for key in (evidence_identity(item), provenance_family(item))
                        if key
                    ],
                }
            )
            state.add_evidence(
                title,
                queries=queries if analysis_title == title else [analysis_title],
                notes=notes,
                source_count=len(_dedup_evidence(evidence)),
                status="gap_closed" if title.strip().lower() in gap_tasks else "researched",
                think_tokens=budget.max_think_tokens,
            )
            section_outputs[-1]["_matrix_index"] = len(state.evidence_matrix) - 1
            if analysis_title != title:
                state.evidence_matrix[-1].subtask = analysis_title
                state.plan = [analysis_title if item == title else item for item in state.plan]
            self._emit_budget(
                rid,
                sink,
                mode=request.mode,
                used=state.searches_used,
                maximum=max_rounds,
                stage="analyst",
                gap_rounds=state.gap_rounds,
                pages_fetched=state.pages_fetched,
                remaining_searches=max(0, max_rounds - state.searches_used),
                search_tier=_latest_external_tier(ctx.retrieval_outcomes),
                degraded_fetch=False,
                recovery_searches_max=1,
                recovery_searches_used=ctx.recovery_searches_used,
            )
            if deadline.expired(reserve_s=retrieval.reserve_total_s):
                if not collected:
                    self._raise_timeout()
                ctx.retrieval_rejection_reason_codes.add("retrieval_stopped_for_writer_reserve")
                self._deep_checkpoint_boundary(
                    "round_complete",
                    request=request,
                    rid=rid,
                    sink=sink,
                    ctx=ctx,
                    state=state,
                    section_outputs=section_outputs,
                    collected=collected,
                    collected_rankings=collected_rankings,
                    gap_tasks=gap_tasks,
                )
                break
            if state.budget_exhausted():
                self._deep_checkpoint_boundary(
                    "round_complete",
                    request=request,
                    rid=rid,
                    sink=sink,
                    ctx=ctx,
                    state=state,
                    section_outputs=section_outputs,
                    collected=collected,
                    collected_rankings=collected_rankings,
                    gap_tasks=gap_tasks,
                )
                break
            if len(state.plan) >= max_rounds:
                self._deep_checkpoint_boundary(
                    "round_complete",
                    request=request,
                    rid=rid,
                    sink=sink,
                    ctx=ctx,
                    state=state,
                    section_outputs=section_outputs,
                    collected=collected,
                    collected_rankings=collected_rankings,
                    gap_tasks=gap_tasks,
                )
                continue
            self._check_cancel(rid)
            self._emit(rid, "node_status", {"node": "reflector", "state": "running"}, sink)
            self._check_deadline(deadline, reserve_s=retrieval.reserve_total_s)
            with use_provider_call_budget(
                deadline.provider_budget(
                    phase_reserve_s=retrieval.reserve_total_s,
                    retrieval_budget=retrieval,
                    interrupt_check=self._stream_cancel_checker(rid, deadline),
                )
            ):
                new_gaps = self._detect_gaps(request.query, section_outputs, budget, ctx)
            added_gaps = state.apply_gap_analysis(new_gaps, max_new=max_rounds - state.searches_used)
            gap_tasks.update(gap.strip().lower() for gap in added_gaps)
            self._emit_budget(
                rid,
                sink,
                mode=request.mode,
                used=state.searches_used,
                maximum=max_rounds,
                stage="reflector",
                gap_rounds=state.gap_rounds,
                pages_fetched=state.pages_fetched,
                remaining_searches=max(0, max_rounds - state.searches_used),
                recovery_searches_max=1,
                recovery_searches_used=ctx.recovery_searches_used,
            )
            self._deep_checkpoint_boundary(
                "round_complete",
                request=request,
                rid=rid,
                sink=sink,
                ctx=ctx,
                state=state,
                section_outputs=section_outputs,
                collected=collected,
                collected_rankings=collected_rankings,
                gap_tasks=gap_tasks,
            )

        self._check_cancel(rid)
        if deadline.expired() and not collected:
            self._raise_timeout()
        timed_out = timed_out or deadline.expired()
        self._emit(rid, "node_status", {"node": "writer", "state": "running"}, sink)
        merged_evidence = _fuse_evidence_rankings(
            collected_rankings,
            budget.top_k,
            preserve_ranking_diversity=True,
            merge_complementary_text=True,
        )
        merged_evidence = self._prepare_adopted_evidence(
            merged_evidence,
            intent=intent,
            request=request,
            budget=budget,
            rid=rid,
            sink=sink,
            ctx=ctx,
            deadline=deadline,
            stage="deep_final",
        )
        merged_evidence, temporal = self._temporal_assurance(
            request,
            budget,
            rid,
            sink,
            ctx,
            merged_evidence,
            plan_id="temporal",
            deadline=deadline,
        )
        timed_out = timed_out or deadline.expired()
        merged_evidence = self._prepare_adopted_evidence(
            merged_evidence,
            intent=intent,
            request=request,
            budget=budget,
            rid=rid,
            sink=sink,
            ctx=ctx,
            deadline=deadline,
            stage="deep_post_temporal",
        )
        merged_evidence, temporal, conflict_analysis = self._final_evidence_snapshot(
            request,
            merged_evidence,
            rid=rid,
            sink=sink,
            ctx=ctx,
            deadline=deadline,
        )
        self._deep_checkpoint_boundary(
            "final_evidence_ready",
            request=request,
            rid=rid,
            sink=sink,
            ctx=ctx,
            state=state,
            section_outputs=section_outputs,
            collected=merged_evidence,
            collected_rankings=[merged_evidence],
            gap_tasks=gap_tasks,
            temporal=temporal,
            conflict_analysis=conflict_analysis,
        )
        if not self._store.fence_deep_final_writer(rid):
            self._check_cancel(rid)
            self._deep_checkpoint_boundary(
                "final_evidence_ready",
                request=request,
                rid=rid,
                sink=sink,
                ctx=ctx,
                state=state,
                section_outputs=section_outputs,
                collected=merged_evidence,
                collected_rankings=[merged_evidence],
                gap_tasks=gap_tasks,
                temporal=temporal,
                conflict_analysis=conflict_analysis,
            )
            raise BaselineError(
                ErrorCode.RUN_NOT_PAUSABLE,
                "Deep run could not enter the final writer fence",
            )
        state_snapshot = _filter_research_state_to_final_evidence(
            state.to_dict(),
            section_outputs,
            merged_evidence,
        )
        writer_context = "\n\n".join(
            part for part in (ctx.conversation_context, render_evidence_matrix(state_snapshot["evidence_matrix"])) if part
        )
        answer, merged_evidence, temporal, conflict_analysis = self._compose_with_repair(
            request=request,
            budget=budget,
            rid=rid,
            sink=sink,
            ctx=ctx,
            deadline=deadline,
            evidence=merged_evidence,
            temporal=temporal,
            conflict_analysis=conflict_analysis,
            structured=True if intent.output_shape == "default" else False,
            context=writer_context,
            base_trusted_instruction=(
                "Synthesize the researched sections into one cohesive, professional report."
                if not timed_out
                else "Synthesize the available researched sections into a partial report because the run timed out."
            ),
        )
        extra = {
            "sections": _filter_sections_to_final_sources(section_outputs, merged_evidence),
            "sources": _sources(merged_evidence),
            "gap_rounds": state.gap_rounds,
            "budget_exhausted": state.budget_exhausted(),
            "research_state": state_snapshot,
            "evidence_matrix": state_snapshot["evidence_matrix"],
            "retrieval_outcomes": [item.to_dict() for item in ctx.retrieval_outcomes],
            "degraded_reasons": list(dict.fromkeys(ctx.degraded_reasons)),
            "temporal_coverage": temporal.to_dict(),
        }
        if timed_out:
            extra["completion_reason"] = "timeout"
        return RunOutcome(
            run_id=rid, mode=request.mode.value, answer=answer,
            status=terminal_status(timed_out=timed_out, degraded_reasons=ctx.degraded_reasons),
            extra=extra,
        )

    def _detect_gaps(self, query: str, sections: list[dict], budget: Budget, ctx: _RunContext) -> list[str]:
        """Ask the LLM which sub-questions remain unanswered; empty list when none."""

        draft = "\n".join(f"- {s['title']}" for s in sections) or "(no sections yet)"
        try:
            result = ctx.llm.complete(
                [{"role": "user", "content": _GAP_PROMPT.format(query=query, draft=draft)}],
                temperature=0.3, max_tokens=budget.max_think_tokens,
            )
        except BaselineError as error:
            if error.code in {ErrorCode.RUN_TIMEOUT, ErrorCode.RUN_CANCELLED}:
                raise
            return []
        except Exception:  # noqa: BLE001 - gaps are best-effort
            return []
        gaps: list[str] = []
        seen: set[str] = set()
        for line in result.text.splitlines():
            # Strip only a leading list marker / enumeration and a trailing period,
            # so legitimate titles keep their digits (e.g. "5G", "Market size 2025").
            cleaned = re.sub(r"^[\s\-*\d.]+", "", line).strip().rstrip(".").strip()
            key = cleaned.lower()
            if len(cleaned) > 4 and key not in _GAP_SENTINELS and key not in seen:
                seen.add(key)
                gaps.append(cleaned)
        return gaps[:3]

    def _run_council(
        self,
        request: RunRequest,
        budget: Budget,
        rid: str,
        sink: EventSink | None,
        ctx: _RunContext,
        deadline: _RunDeadline,
    ) -> RunOutcome:
        intent = self._intent(ctx)
        retrieval = ctx.retrieval_budget or retrieval_budget_for(request.mode)
        self._check_deadline(deadline, reserve_s=retrieval.reserve_total_s)
        self._emit(rid, "node_status", {"node": "search", "state": "running"}, sink)
        evidence = self._retrieve_evidence_plan(
            intent.research_focus,
            request,
            budget,
            rid,
            sink,
            ctx,
            plan_id="primary",
            deadline=deadline,
        )
        ctx.ordinary_searches_used = 1
        self._emit_budget(
            rid,
            sink,
            mode=request.mode,
            used=1,
            maximum=budget.max_searches,
            stage="search",
            recovery_searches_max=0,
            recovery_searches_used=0,
        )
        self._check_cancel(rid)
        self._emit(rid, "node_status", {"node": "retrieve", "state": "running"}, sink)
        evidence, _ = self._temporal_assurance(
            request,
            budget,
            rid,
            sink,
            ctx,
            evidence,
            plan_id="temporal",
            deadline=deadline,
        )
        self._check_deadline(deadline, reserve_s=retrieval.reserve_total_s)
        with use_provider_call_budget(
            deadline.provider_budget(
                phase_reserve_s=retrieval.reserve_total_s,
                retrieval_budget=retrieval,
                interrupt_check=self._network_interrupt_checker(rid, deadline),
                max_attempts=2,
            )
        ):
            evidence, temporal, conflict_analysis = self._final_evidence_snapshot(
                request,
                evidence,
                rid=rid,
                sink=sink,
                ctx=ctx,
                deadline=deadline,
            )
        trust_constraints = build_council_trust_constraints(
            temporal,
            conflict_analysis[0],
            conflict_analysis[1],
            evidence=evidence,
        )
        members = ctx.members or [ctx.llm]

        def _council_stage(stage: str) -> None:
            """Emit truthful stage events as run_council actually enters each phase."""
            payload: dict = {"node": stage, "state": "running"}
            if stage == "drafts":
                payload["members"] = len(members)
            self._emit(rid, "node_status", payload, sink)
            self._emit_budget(
                rid,
                sink,
                mode=request.mode,
                used=1,
                maximum=budget.max_searches,
                stage=stage,
                recovery_searches_max=0,
                recovery_searches_used=0,
            )

        ctx.writer_started_at = ctx.writer_started_at or time.time()
        writer_started_at = time.time()
        writer_remaining_before = deadline.remaining_s()
        writer_status = "failed"
        try:
            with use_provider_call_budget(
                deadline.provider_budget(
                    phase_reserve_s=retrieval.finalize_reserve_s + retrieval.cleanup_reserve_s,
                    retrieval_budget=retrieval,
                    interrupt_check=self._network_interrupt_checker(rid, deadline),
                    max_attempts=max(2, len(members)),
                )
            ):
                result = run_council(
                    members, request.query, evidence,
                    max_output_tokens=budget.max_output_tokens, concurrency=budget.concurrency,
                    judge=None if request.council_strategy == "parallel_vote" else ctx.llm,
                    temperature=ctx.llm_params.temperature, thinking_budget=ctx.llm_params.thinking_budget,
                    strategy=request.council_strategy,
                    on_stage=_council_stage,
                    on_consensus_delta=(ctx.answer_stream.accept if ctx.answer_stream is not None else None),
                    on_consensus_reset=(
                        (lambda reason: ctx.answer_stream.reset(reason))
                        if ctx.answer_stream is not None
                        else None
                    ),
                    context=ctx.conversation_context,
                    trust_constraints=trust_constraints,
                )
            writer_status = "completed"
        finally:
            self._append_phase_metric(
                ctx,
                stage="council_writer",
                query=request.query,
                provider=str(getattr(ctx.llm, "name", "") or "") or None,
                started_at=writer_started_at,
                finished_at=time.time(),
                status=writer_status,
                deadline=deadline,
                remaining_before_s=writer_remaining_before,
            )
        if ctx.answer_stream is not None:
            ctx.answer_stream.finish_generation()
            ctx.answer_stream.mark_sync_fallback(result.stream_sync_fallback_used)
        self._finalize(
            result.consensus,
            Mode.COUNCIL,
            rid,
            evidence,
            ctx=ctx,
            sink=sink,
            conflict_analysis=conflict_analysis,
            deadline=deadline,
        )
        if result.member_fallback_used:
            self._add_degraded_reason(ctx, "council_member_fallback")
            ctx.verification_reasons.append("council_member_fallback")
        enrich_council_structure(result.consensus)
        self._check_cancel(rid)
        self._check_deadline(deadline, reserve_s=retrieval.cleanup_reserve_s)
        return RunOutcome(
            run_id=rid, mode=request.mode.value, answer=result.consensus,
            status=terminal_status(timed_out=False, degraded_reasons=ctx.degraded_reasons),
            extra={
                "member_fallback_used": result.member_fallback_used,
                "confidence": result.confidence,
                "breakdown": result.breakdown,
                "consensus_strength": result.consensus_strength,
                "members": [m.to_dict() for m in result.members],
                "critiques": result.critiques,
                **({"member_failures": result.member_failures} if result.member_failures else {}),
                "sources": _sources(evidence),
                "retrieval_outcomes": [item.to_dict() for item in ctx.retrieval_outcomes],
                "degraded_reasons": list(dict.fromkeys(ctx.degraded_reasons)),
                "temporal_coverage": temporal.to_dict(),
            },
        )

    # --- helpers --------------------------------------------------------------

    @staticmethod
    def _intent(ctx: _RunContext) -> ResearchIntent:
        if ctx.research_intent is None:
            raise BaselineError(ErrorCode.INTERNAL_ERROR, "Research intent was not initialized.")
        return ctx.research_intent

    def _assess_and_record(
        self,
        intent: ResearchIntent,
        evidence: list[EvidenceItem],
        rid: str,
        sink: EventSink | None,
        ctx: _RunContext,
        *,
        stage: str,
    ) -> EvidenceAssessment:
        assessment = assess_evidence(intent, evidence)
        for outcome in assessment.outcomes:
            ctx.evidence_eligibility[outcome.evidence_id] = outcome
            self._emit(
                rid,
                "evidence_eligibility",
                {**outcome.to_dict(), "stage": stage},
                sink,
            )
        return assessment

    @staticmethod
    def _raise_eligibility_required(assessment: EvidenceAssessment) -> None:
        raise BaselineError(
            ErrorCode.GROUNDING_REQUIRED,
            "No sufficiently relevant and citable evidence was available for this research query.",
            details={
                "reason": assessment.trigger or "zero_citable_evidence",
                "eligible_count": len(assessment.adopted),
                "evidence_count": len(assessment.outcomes),
            },
        )

    def _prepare_adopted_evidence(
        self,
        evidence: list[EvidenceItem],
        *,
        intent: ResearchIntent,
        request: RunRequest,
        budget: Budget,
        rid: str,
        sink: EventSink | None,
        ctx: _RunContext,
        deadline: _RunDeadline,
        stage: str,
    ) -> list[EvidenceItem]:
        retrieval = ctx.retrieval_budget or retrieval_budget_for(request.mode)
        self._check_deadline(deadline, reserve_s=retrieval.reserve_total_s)
        assessment = self._assess_and_record(intent, evidence, rid, sink, ctx, stage=stage)
        # Relevance diagnostics inform confidence; they never veto delivery.
        return list(assessment.adopted)

    def _run_retrieval_remediation(
        self,
        evidence: list[EvidenceItem],
        assessment: EvidenceAssessment,
        *,
        intent: ResearchIntent,
        request: RunRequest,
        budget: Budget,
        rid: str,
        sink: EventSink | None,
        ctx: _RunContext,
        deadline: _RunDeadline,
        source_generation_id: str | None,
    ) -> list[EvidenceItem]:
        if ctx.remediation.attempted:
            self._raise_eligibility_required(assessment)
        self._check_cancel(rid)
        if deadline.expired():
            ctx.remediation = RemediationOutcome(
                trigger=assessment.trigger,
                focused_query=intent.research_focus,
                status="timeout",
            )
            self._emit(rid, "remediation_outcome", ctx.remediation.to_dict(), sink)
            self._raise_timeout()

        recovery_query = focused_recovery_query(intent, assessment.trigger)
        excluded_provenance_roots = tuple(
            dict.fromkeys(
                (
                    *assessment.rejected_provenance_roots,
                    *sorted(ctx.unready_external_provenance_roots),
                )
            )
        )
        ctx.remediation = started_remediation(
            trigger=assessment.trigger or "prewriter_irrelevant",
            action="retrieve_and_rewrite",
            focused_query=recovery_query,
            excluded_evidence_ids=assessment.rejected_ids,
            excluded_provenance_roots=excluded_provenance_roots,
            source_generation_id=source_generation_id,
        )
        ctx.recovery_searches_used = 1
        self._emit(
            rid,
            "node_status",
            {
                "node": "source_recovery",
                "state": "running",
                "message": "Verifying better sources…",
            },
            sink,
        )
        self._emit_budget(
            rid,
            sink,
            mode=request.mode,
            used=ctx.ordinary_searches_used,
            maximum=budget.max_searches,
            stage="source_recovery",
            recovery_searches_max=1,
            recovery_searches_used=1,
        )
        try:
            recovered = self._retrieve_evidence_plan(
                recovery_query,
                request,
                budget,
                rid,
                sink,
                ctx,
                plan_id="recovery-1",
                recovery=True,
                excluded_evidence_ids=assessment.rejected_ids,
                excluded_provenance_roots=excluded_provenance_roots,
                        relevance_query=intent.research_focus,
                        deadline=deadline,
            )
            rejected = set(assessment.rejected_ids)
            retained = [item for item in evidence if evidence_identity(item) not in rejected]
            combined = _fuse_evidence_rankings(
                [ranking for ranking in (retained, recovered) if ranking],
                budget.top_k,
            )
            recovered_assessment = self._assess_and_record(
                intent,
                combined,
                rid,
                sink,
                ctx,
                stage="recovery-1",
            )
            ctx.remediation = replace(
                ctx.remediation,
                lane_ids=("recovery-1:local", "recovery-1:memory", "recovery-1:external"),
                new_evidence_ids=tuple(evidence_identity(item) for item in recovered),
                status="completed" if recovered_assessment.trigger is None else "exhausted",
                finished_at=time.time(),
            )
            if recovered_assessment.trigger is not None:
                self._raise_eligibility_required(recovered_assessment)
            self._emit(rid, "remediation_outcome", ctx.remediation.to_dict(), sink)
            return list(recovered_assessment.adopted)
        except BaselineError as error:
            status = (
                "cancelled"
                if error.code is ErrorCode.RUN_CANCELLED
                else "timeout"
                if error.code is ErrorCode.RUN_TIMEOUT
                else "exhausted"
                if error.code is ErrorCode.GROUNDING_REQUIRED
                else "failed"
            )
            ctx.remediation = replace(ctx.remediation, status=status, finished_at=time.time())
            self._emit(rid, "remediation_outcome", ctx.remediation.to_dict(), sink)
            raise

    def _final_evidence_snapshot(
        self,
        request: RunRequest,
        evidence: list[EvidenceItem],
        *,
        rid: str,
        sink: EventSink | None,
        ctx: _RunContext,
        deadline: _RunDeadline,
    ) -> tuple[list[EvidenceItem], TemporalCoverage, tuple[list, list[dict]]]:
        intent = self._intent(ctx)
        assessment = self._assess_and_record(
            intent,
            evidence,
            rid,
            sink,
            ctx,
            stage="final_snapshot",
        )
        if assessment.trigger:
            ctx.verification_reasons.append(str(assessment.trigger))
        candidates = list(assessment.adopted)
        final_evidence = apply_citation_limit(intent, candidates, conflicts_present=False)
        final_evidence = self._materialize_transient_external_evidence(
            final_evidence, adoption_run_id=rid, ctx=ctx, deadline=deadline,
        )
        self._start_verification(request, final_evidence, ctx)
        conflict_analysis = ([], [])
        temporal_intent, requested_as_of = detect_temporal_intent(request.query)
        temporal = assess_temporal_coverage(
            intent=temporal_intent,
            requested_as_of=requested_as_of,
            evidence=final_evidence,
            checked_at=time.time(),
        )
        self._replace_temporal_reasons(ctx, temporal)
        self._apply_final_locale_degradation(ctx, final_evidence)
        return final_evidence, temporal, conflict_analysis

    def _materialize_transient_external_evidence(
        self,
        evidence: list[EvidenceItem],
        *,
        adoption_run_id: str,
        ctx: _RunContext,
        deadline: _RunDeadline,
    ) -> list[EvidenceItem]:
        """Index only finally adopted safe snippets, never unused search hits."""

        materialized: list[EvidenceItem] = []
        for item in evidence:
            if (
                item.origin != "external"
                or item.chunk_id >= 0
                or not str(item.url or "").startswith(("http://", "https://"))
                or not item.text.strip()
            ):
                materialized.append(item)
                continue
            retrieval = ctx.retrieval_budget or retrieval_budget_for(Mode.FAST)
            self._check_deadline(deadline, reserve_s=retrieval.reserve_total_s)
            provenance_root = provenance_family(item)
            prepared_entry = ctx.prepared_external_documents.get(provenance_root)
            prepared = prepared_entry[0] if prepared_entry is not None else None
            selected_piece_index = prepared_entry[1] if prepared_entry is not None else 0
            prepare_document = getattr(self._indexer, "prepare_document", None)
            commit_document = getattr(self._indexer, "commit_prepared_document", None)
            if prepared is None and callable(prepare_document):
                prepared = self._prepare_transient_external_document(
                    prepare_document,
                    Document(
                        url=str(item.url),
                        title=item.title or str(item.url),
                        text=item.text,
                        content_type="web",
                        metadata={"content_basis": "snippet"},
                    )
                )
                selected_piece_index = 0
            self._check_deadline(deadline, reserve_s=retrieval.reserve_total_s)
            if prepared is not None and callable(commit_document):
                stats = commit_document(prepared, adoption_run_id=adoption_run_id)
                document_id = str(stats.get("document_id") or item.document_id)
                chunk_ids = list(stats.get("chunk_ids") or [])
                chunk_id = (
                    int(chunk_ids[selected_piece_index])
                    if 0 <= selected_piece_index < len(chunk_ids)
                    else (int(chunk_ids[0]) if chunk_ids else -1)
                )
            else:
                document_id = self._store.upsert_document(
                    url=str(item.url),
                    title=item.title or str(item.url),
                    text=item.text,
                    content_type="web",
                    metadata={"content_basis": "snippet"},
                    quarantine_library_candidate=True,
                )
                chunk_id = self._store.add_chunk(
                    document_id=document_id,
                    chunk_index=0,
                    text=item.text,
                )
            materialized.append(
                replace(
                    item,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    artifact_id=document_id,
                )
            )
        return materialized

    @staticmethod
    def _prepare_transient_external_document(
        prepare_document: Callable[..., object],
        document: Document,
    ) -> object:
        """Prepare citable text without spending the writer reserve on optional vectors."""

        try:
            supports_embedding_control = (
                "include_embeddings" in inspect.signature(prepare_document).parameters
            )
        except (TypeError, ValueError):
            supports_embedding_control = False
        if supports_embedding_control:
            return prepare_document(document, include_embeddings=False)
        return prepare_document(document)

    @staticmethod
    def _replace_temporal_reasons(ctx: _RunContext, coverage: TemporalCoverage) -> None:
        if ctx.temporal_reason_codes:
            ctx.degraded_reasons = [
                reason for reason in ctx.degraded_reasons if reason not in ctx.temporal_reason_codes
            ]
        current = set(temporal_degraded_reasons(coverage))
        ctx.temporal_reason_codes = current
        for reason in current:
            if reason not in ctx.degraded_reasons:
                ctx.degraded_reasons.append(reason)

    @staticmethod
    def _apply_final_locale_degradation(ctx: _RunContext, evidence: list[EvidenceItem]) -> None:
        ctx.degraded_reasons = [
            reason for reason in ctx.degraded_reasons if reason != "search_locale_partially_applied"
        ]
        final_ids = {evidence_identity(item) for item in evidence}
        for outcome in ctx.retrieval_outcomes:
            if (
                outcome.lane == "external"
                and outcome.locale
                and outcome.locale.get("ignored_fields")
                and final_ids.intersection(outcome.evidence_ids)
            ):
                ctx.degraded_reasons.append("search_locale_partially_applied")
                return

    def _compose_with_repair(
        self,
        *,
        request: RunRequest,
        budget: Budget,
        rid: str,
        sink: EventSink | None,
        ctx: _RunContext,
        deadline: _RunDeadline,
        evidence: list[EvidenceItem],
        temporal: TemporalCoverage,
        conflict_analysis: tuple[list, list[dict]],
        structured: bool | str,
        context: str,
        base_trusted_instruction: str = "",
    ) -> tuple[GroundedAnswer, list[EvidenceItem], TemporalCoverage, tuple[list, list[dict]]]:
        intent = self._intent(ctx)
        repair_instruction = ""

        def compose_once() -> GroundedAnswer:
            retrieval = ctx.retrieval_budget or retrieval_budget_for(request.mode)
            self._check_deadline(deadline, reserve_s=retrieval.reserve_total_s)
            if ctx.writer_started_at is None:
                ctx.writer_started_at = time.time()
            phase_started_at = time.time()
            phase_remaining_before = deadline.remaining_s()
            phase_status = "failed"
            try:
                with use_provider_call_budget(
                    deadline.provider_budget(
                        phase_reserve_s=retrieval.finalize_reserve_s + retrieval.cleanup_reserve_s,
                        retrieval_budget=retrieval,
                        interrupt_check=self._stream_cancel_checker(rid, deadline),
                        max_attempts=3,
                    )
                ):
                    answer = self._compose_streamed(
                        ctx,
                        request.query,
                        evidence,
                        max_tokens=budget.max_output_tokens,
                        instruction=output_instruction(intent),
                        structured=structured,
                        temperature=ctx.llm_params.temperature,
                        thinking_budget=ctx.llm_params.thinking_budget,
                        trusted_instruction=_combine_instructions(
                            base_trusted_instruction,
                            _temporal_instruction(temporal),
                            _conflict_instruction(conflict_analysis),
                            repair_instruction,
                        ),
                        context=context,
                    )
                phase_status = "completed"
            finally:
                self._append_phase_metric(
                    ctx,
                    stage="final_writer",
                    query=request.query,
                    provider=str(getattr(ctx.llm, "name", "") or "") or None,
                    started_at=phase_started_at,
                    finished_at=time.time(),
                    status=phase_status,
                    deadline=deadline,
                    remaining_before_s=phase_remaining_before,
                )
            if not output_shape_valid(intent, answer.text):
                ctx.verification_reasons.append("output_shape_invalid")
            self._finalize(
                answer,
                request.mode,
                rid,
                evidence,
                ctx=ctx,
                sink=sink,
                conflict_analysis=conflict_analysis,
                deadline=deadline,
            )
            return answer

        return compose_once(), evidence, temporal, conflict_analysis

    @staticmethod
    def _t48_audit_metadata(ctx: _RunContext) -> dict[str, object]:
        counts = {"eligible": 0, "rejected": 0, "unknown": 0}
        for item in ctx.evidence_eligibility.values():
            counts[item.decision] += 1
        return {
            "t48": {
                "research_intent": ctx.research_intent.to_dict() if ctx.research_intent else None,
                "eligibility_counts": counts,
                "remediation": ctx.remediation.to_dict(),
                "recovery_searches_max": ctx.recovery_searches_max,
                "recovery_searches_used": ctx.recovery_searches_used,
            }
        }

    @staticmethod
    def _compose_streamed(
        ctx: _RunContext,
        query: str,
        evidence: list[EvidenceItem],
        **kwargs,
    ) -> GroundedAnswer:
        """Stream one final writer draft and preserve explicit failure/reset semantics."""

        stream = ctx.answer_stream
        if stream is None:
            answer = ctx.composer.compose(query, evidence, **kwargs)
            ctx.writer_diagnostics = {
                "classification": "completed",
                "visible_chars": len(answer.text),
                "fallback_scope": "none",
                "attempts": [],
            }
            return answer
        try:
            generation_id = stream.generation_id
            answer, metadata = ctx.composer.compose_stream(
                query,
                evidence,
                on_delta=stream.accept,
                **kwargs,
            )
            stream.finish_generation()
            stream.mark_sync_fallback(metadata.sync_fallback_used)
            ctx.writer_diagnostics = _merge_writer_diagnostics(
                ctx.writer_diagnostics,
                _writer_meta_payload(metadata, generation_id=generation_id),
            )
            return answer
        except BaselineError as error:
            ctx.writer_diagnostics = _merge_writer_diagnostics(
                ctx.writer_diagnostics,
                _writer_meta_payload(
                    ctx.composer.last_stream_meta,
                    generation_id=stream.generation_id,
                ),
            )
            raw_error_diagnostics = error.details.get("writer_diagnostics")
            if isinstance(raw_error_diagnostics, dict):
                safe_error_diagnostics = _safe_writer_diagnostics(raw_error_diagnostics)
                safe_error_diagnostics.pop("attempts", None)
                ctx.writer_diagnostics.update(safe_error_diagnostics)
            if error.code is ErrorCode.RUN_TIMEOUT:
                ctx.writer_diagnostics["classification"] = "deadline_exceeded"
            if stream.draft_active:
                if error.code is ErrorCode.RUN_TIMEOUT:
                    stream.reset("deadline_exceeded")
                elif error.code not in {ErrorCode.RUN_CANCELLED, ErrorCode.STREAM_LIMIT_EXCEEDED}:
                    stream.reset("provider_stream_failed")
            raise
        except Exception as error:  # noqa: BLE001 - provider details stay internal
            ctx.writer_diagnostics = _merge_writer_diagnostics(
                ctx.writer_diagnostics,
                _writer_meta_payload(
                    ctx.composer.last_stream_meta,
                    generation_id=stream.generation_id,
                ),
            )
            ctx.writer_diagnostics["classification"] = "provider_error"
            if stream.draft_active:
                stream.reset("provider_stream_failed")
            raise BaselineError(ErrorCode.UPSTREAM_ERROR, "LLM provider stream failed.") from error

    def _retrieve_evidence_plan(
        self,
        query: str,
        request: RunRequest,
        budget: Budget,
        rid: str,
        sink: EventSink | None,
        ctx: _RunContext,
        *,
        plan_id: str,
        recovery: bool = False,
        excluded_evidence_ids: tuple[str, ...] = (),
        excluded_provenance_roots: tuple[str, ...] = (),
        relevance_query: str | None = None,
        deadline: _RunDeadline | None = None,
        external_search_handle: _ExternalSearchHandle | None = None,
        external_search_started: bool = False,
    ) -> list[EvidenceItem]:
        """Plan all three lanes first, then execute them in fixed isolation order."""

        lanes = ("local", "memory", "external")
        self._emit(
            rid,
            "retrieval_planned",
            {
                "query": query,
                "lanes": [
                    {"lane_id": f"{plan_id}:{lane}", "lane": lane, "query": query}
                    for lane in lanes
                ],
            },
            sink,
        )
        external_handle = external_search_handle
        if not external_search_started:
            external_handle = self._begin_external_search(
                query,
                request=request,
                ctx=ctx,
                rid=rid,
                recovery=recovery,
                excluded_provenance_roots=excluded_provenance_roots,
                relevance_query=relevance_query,
                deadline=deadline,
            )
        rankings: list[list[EvidenceItem]] = [list(ctx.regeneration_evidence)] if ctx.regeneration_evidence else []
        deadline_exhausted = False
        for lane in lanes:
            self._check_cancel(rid)
            started_at = time.time()
            remaining_before_s = deadline.remaining_s() if deadline is not None else None
            lane_evidence: list[EvidenceItem] = []
            provider: str | None = None
            tier: str | None = None
            error_code: str | None = None
            reason: str | None = None
            locale: dict | None = None
            search_status: dict[str, object] = {}
            status = "empty"
            try:
                retrieval = ctx.retrieval_budget or retrieval_budget_for(request.mode)
                if deadline is not None and deadline.expired(reserve_s=retrieval.reserve_total_s):
                    raise BaselineError(
                        ErrorCode.RUN_TIMEOUT,
                        f"Run deadline was exceeded before the {lane} evidence lane.",
                    )
                phase_scope = (
                    use_provider_call_budget(
                        deadline.provider_budget(
                            phase_reserve_s=retrieval.reserve_total_s,
                            retrieval_budget=retrieval,
                            interrupt_check=self._stream_cancel_checker(rid, deadline),
                            max_attempts=2,
                        )
                    )
                    if deadline is not None
                    else nullcontext()
                )
                with phase_scope:
                    if lane == "local":
                        retrieve_local = getattr(self._library_retriever, "retrieve_evidence", None)
                        if callable(retrieve_local):
                            lane_evidence = list(retrieve_local(query, top_k=budget.top_k))
                        else:
                            reason = "retriever_unavailable"
                    elif lane == "memory":
                        recall_evidence = getattr(self._memory, "recall_evidence", None)
                        if callable(recall_evidence):
                            lane_evidence = list(
                                recall_evidence(
                                    session_key=request.session_id or "",
                                    workspace_key=request.workspace_key,
                                    user_key=request.user_key,
                                    limit=budget.top_k,
                                    query=query,
                                )
                            )
                        else:
                            reason = "retriever_unavailable"
                    else:
                        prior_evidence = _fuse_evidence_rankings(
                            [ranking for ranking in rankings if ranking],
                            budget.top_k,
                        )
                        prior_assessment = assess_evidence(
                            _retrieval_intent(self._intent(ctx), relevance_query),
                            prior_evidence,
                        )
                        external = self._retrieve_external_evidence(
                            query,
                            request=request,
                            budget=budget,
                            rid=rid,
                            sink=sink,
                            ctx=ctx,
                            plan_id=plan_id,
                            recovery=recovery,
                            excluded_provenance_roots=excluded_provenance_roots,
                            relevance_query=relevance_query,
                            deadline=deadline,
                            search_handle=external_handle,
                            prior_ready_count=len(_dedup_evidence(prior_assessment.adopted)),
                        )
                        lane_evidence = list(external["evidence"])
                        provider = str(external.get("provider") or "") or None
                        tier = str(external.get("tier") or "") or None
                        reason = str(external.get("reason") or "") or None
                        raw_locale = external.get("locale")
                        if isinstance(raw_locale, dict):
                            locale = dict(raw_locale)
                        search_status = dict(external.get("status") or {})
                    raw_locale = search_status.get("locale")
                    if lane == "external" and isinstance(raw_locale, dict):
                        locale = dict(raw_locale)
                    elif lane == "external" and request.effective_locale:
                        ignored_fields = (
                            [field for field in ("country", "search_lang", "ui_language") if field in request.requested_locale]
                            if tier == "indexed_external"
                            else []
                        )
                        locale = {
                            "requested": dict(request.requested_locale),
                            "effective": dict(request.effective_locale),
                            "applied_fields": [],
                            "ignored_fields": ignored_fields,
                            "reason_codes": ["indexed_external_locale_not_applied"] if ignored_fields else [],
                        }
                    if lane == "external" and search_status.get("live_providers_down"):
                        reason = "provider_degraded"
                        self._add_degraded_reason(ctx, "search_provider_degraded")
                    if deadline is not None:
                        self._check_deadline(deadline, reserve_s=retrieval.reserve_total_s)
                lane_evidence = _exclude_evidence(
                    lane_evidence,
                    evidence_ids=excluded_evidence_ids,
                    provenance_roots=excluded_provenance_roots,
                )
                status = "completed" if lane_evidence else "empty"
            except BaselineError as error:
                if error.code is ErrorCode.RUN_CANCELLED:
                    raise
                _LOG.warning(
                    "evidence_lane_baseline_error",
                    run_id=rid,
                    plan_id=plan_id,
                    lane=lane,
                    error_code=error.code.value,
                )
                status = "timeout" if error.code is ErrorCode.RUN_TIMEOUT else "failed"
                deadline_exhausted = deadline_exhausted or error.code is ErrorCode.RUN_TIMEOUT
                error_code = error.code.value
                reason = "lane_timeout" if status == "timeout" else "lane_failed"
            except Exception as error:
                _LOG.warning(
                    "evidence_lane_failed",
                    run_id=rid,
                    plan_id=plan_id,
                    lane=lane,
                    error_type=error.__class__.__name__,
                )
                status = "failed"
                error_code = ErrorCode.UPSTREAM_ERROR.value
                reason = "lane_failed"
            finished_at = time.time()
            lane_evidence = [
                replace(
                    item,
                    retrieved_at=(
                        item.retrieved_at if item.retrieved_at is not None else finished_at
                    ),
                )
                for item in lane_evidence
            ]
            outcome = RetrievalOutcome(
                lane_id=f"{plan_id}:{lane}",
                lane=lane,
                query=query,
                attempted=True,
                status=status,
                provider=provider,
                search_provider=(provider if lane == "external" else None),
                fetch_providers=tuple(
                    dict.fromkeys(
                        item.fetch_provider
                        for item in lane_evidence
                        if lane == "external" and item.fetch_provider
                    )
                ),
                providers_attempted=tuple(search_status.get("providers_attempted") or ()),
                providers_skipped=tuple(search_status.get("skipped_providers") or ()),
                provider_failure_reasons=dict(search_status.get("provider_failure_reasons") or {}),
                tier=tier,
                hit_count=len(lane_evidence),
                evidence_ids=tuple(evidence_identity(item) for item in lane_evidence),
                error_code=error_code,
                reason=reason,
                locale=locale,
                search_options=(
                    dict(search_status.get("search_options"))
                    if lane == "external" and isinstance(search_status.get("search_options"), dict)
                    else None
                ),
                started_at=started_at,
                finished_at=finished_at,
            )
            ctx.retrieval_outcomes.append(outcome)
            if status in {"failed", "timeout"}:
                self._add_degraded_reason(ctx, f"retrieval_{lane}_{status}")
            self._emit(rid, "retrieval_outcome", outcome.to_dict(), sink)
            if deadline is not None:
                self._append_phase_metric(
                    ctx,
                    stage="evidence_lane",
                    plan_id=plan_id,
                    lane=lane,
                    query=query,
                    provider=provider,
                    started_at=started_at,
                    finished_at=finished_at,
                    status=status,
                    deadline=deadline,
                    remaining_before_s=remaining_before_s,
                )
            rankings.append(lane_evidence)
        if deadline_exhausted:
            self._close_external_search_handle(external_handle)
            self._raise_timeout()
        self._close_external_search_handle(external_handle)
        kg_evidence = self._retrieve_kg_candidates(
            query,
            request=request,
            rid=rid,
            ctx=ctx,
            excluded_evidence_ids=excluded_evidence_ids,
            excluded_provenance_roots=excluded_provenance_roots,
            deadline=deadline,
        )
        if kg_evidence:
            rankings.append(kg_evidence)
        return _fuse_evidence_rankings(rankings, budget.top_k)

    def _retrieve_kg_candidates(
        self,
        query: str,
        *,
        request: RunRequest,
        rid: str,
        ctx: _RunContext,
        excluded_evidence_ids: tuple[str, ...],
        excluded_provenance_roots: tuple[str, ...],
        deadline: _RunDeadline | None,
    ) -> list[EvidenceItem]:
        """Use the frozen graph only for canonical candidate discovery."""

        snapshot = (
            request.kg_retrieval_snapshot
            if isinstance(request.kg_retrieval_snapshot, dict)
            else {}
        )
        if not ctx.kg_retrieval.requested or not bool(
            snapshot.get("effective", False)
        ):
            return []
        self._check_cancel(rid)
        retrieval = ctx.retrieval_budget or retrieval_budget_for(request.mode)
        if deadline is not None:
            self._check_deadline(deadline, reserve_s=retrieval.reserve_total_s)
        started_at = time.time()
        remaining_before_s = deadline.remaining_s() if deadline is not None else None
        status = "empty"
        try:
            result = self._kg_retriever.retrieve(
                query,
                workspace_id=str(
                    snapshot.get("workspace_id")
                    or request.workspace_key
                    or "default"
                ),
                requested=True,
                mode=request.mode.value,
                snapshot_version_id=str(snapshot.get("version_id") or ""),
                snapshot_signature=str(
                    snapshot.get("canonical_evidence_signature") or ""
                ),
            )
            ctx.kg_retrieval.merge(result.diagnostics)
            evidence = _exclude_evidence(
                result.evidence,
                evidence_ids=excluded_evidence_ids,
                provenance_roots=excluded_provenance_roots,
            )
            evidence = _dedup_evidence(evidence)
            status = "completed" if evidence else "empty"
            for reason in result.diagnostics.reason_codes:
                if reason in _KG_DEGRADED_REASON_CODES:
                    self._add_degraded_reason(
                        ctx,
                        f"kg_retrieval_{reason}",
                    )
            if deadline is not None:
                self._check_deadline(
                    deadline,
                    reserve_s=retrieval.reserve_total_s,
                )
            return evidence
        except BaselineError:
            raise
        except Exception as error:  # noqa: BLE001 - KG is an optional local path
            _LOG.warning(
                "kg_candidate_discovery_failed",
                run_id=rid,
                error_type=error.__class__.__name__,
            )
            ctx.kg_retrieval.add_reason("kg_read_failed")
            self._add_degraded_reason(ctx, "kg_retrieval_kg_read_failed")
            status = "failed"
            return []
        finally:
            if deadline is not None:
                self._append_phase_metric(
                    ctx,
                    stage="kg_candidate_discovery",
                    query=query,
                    started_at=started_at,
                    finished_at=time.time(),
                    status=status,
                    deadline=deadline,
                    remaining_before_s=remaining_before_s,
                )

    def _begin_external_search(
        self,
        query: str,
        *,
        request: RunRequest,
        ctx: _RunContext,
        rid: str,
        recovery: bool,
        excluded_provenance_roots: tuple[str, ...],
        relevance_query: str | None,
        deadline: _RunDeadline | None,
    ) -> _ExternalSearchHandle | None:
        """Start only a proven Store-free network search while local lanes execute."""

        search = self._search
        search_network = getattr(search, "search_network", None)
        if search is None or not callable(search_network):
            return None
        retrieval = ctx.retrieval_budget or retrieval_budget_for(request.mode)
        options = runtime_search_options(request.search_options, request.effective_locale)
        usage_kwargs = _search_usage_kwargs(search, request)
        temporal_intent, _ = detect_temporal_intent(request.query)
        provider_relevance_query = relevance_query or self._intent(ctx).original_query or request.query
        relevance_kwargs = _search_relevance_kwargs(search, provider_relevance_query)
        read_cache = getattr(search, "read_search_cache", None)
        if temporal_intent == "none" and not recovery and not ctx.fresh_search_required and callable(read_cache):
            cached = read_cache(
                query,
                count=retrieval.results_per_query,
                options=options,
                **relevance_kwargs,
                **usage_kwargs,
            )
            if cached is not None:
                return _ExternalSearchHandle(
                    result=cached,
                    started_at=time.time(),
                    remaining_before_s=deadline.remaining_s() if deadline is not None else None,
                )
        if deadline is not None:
            self._check_deadline(deadline, reserve_s=retrieval.reserve_total_s)
            provider_budget = deadline.provider_budget(
                phase_reserve_s=retrieval.reserve_total_s,
                retrieval_budget=retrieval,
                max_attempts=2,
                interrupt_check=self._network_interrupt_checker(rid, deadline),
            )
        else:
            provider_budget = None

        provider_circuit = set(ctx.provider_circuit_open)

        def _search_network_only() -> SearchCallResult:
            with use_provider_call_budget(provider_budget):
                return search_network(
                    query,
                    count=retrieval.results_per_query,
                    options=options,
                    recovery=recovery,
                    excluded_provenance_roots=excluded_provenance_roots,
                    relevance_query=provider_relevance_query,
                    **(
                        {"provider_circuit": provider_circuit}
                        if getattr(search, "supports_search_usage_policy", False)
                        else {}
                    ),
                    **usage_kwargs,
                )

        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="t49-search")
        handle = _ExternalSearchHandle(
            future=executor.submit(_search_network_only),
            executor=executor,
            provider_circuit=provider_circuit,
            started_at=time.time(),
            remaining_before_s=deadline.remaining_s() if deadline is not None else None,
        )
        return handle

    @staticmethod
    def _close_external_search_handle(handle: _ExternalSearchHandle | None) -> None:
        if handle is None or handle.executor is None:
            return
        if handle.future is not None and not handle.future.done():
            handle.future.cancel()
        handle.executor.shutdown(wait=False, cancel_futures=True)
        handle.executor = None

    def _retrieve_external_evidence(
        self,
        query: str,
        *,
        request: RunRequest,
        budget: Budget,
        rid: str,
        sink: EventSink | None,
        ctx: _RunContext,
        plan_id: str,
        recovery: bool,
        excluded_provenance_roots: tuple[str, ...],
        relevance_query: str | None,
        deadline: _RunDeadline | None,
        search_handle: _ExternalSearchHandle | None,
        prior_ready_count: int = 0,
    ) -> dict[str, object]:
        """Search candidates, demand-fetch only the needed pages, then return evidence."""

        retrieval = ctx.retrieval_budget or retrieval_budget_for(request.mode)
        if self._search is None or self._indexer is None:
            return {"evidence": [], "status": {}, "reason": "retriever_unavailable"}
        options = runtime_search_options(request.search_options, request.effective_locale)
        hits, status = self._search_external_candidates(
            query,
            count=retrieval.results_per_query,
            options=options,
            request=request,
            plan_id=plan_id,
            recovery=recovery,
            excluded_provenance_roots=excluded_provenance_roots,
            relevance_query=relevance_query,
            deadline=deadline,
            ctx=ctx,
            rid=rid,
            search_handle=search_handle,
        )
        tier = str(status.get("fallback_tier") or status.get("producer_tier") or "")
        provider = str(
            status.get("selected_provider")
            or status.get("producer_provider")
            or status.get("producer_tier")
            or ""
        )
        for name in status.get("providers_attempted") or []:
            if str(name):
                normalized_name = "firecrawl" if str(name) == "firecrawl_search" else str(name)
                ctx.search_providers.add(normalized_name)
                _append_unique(ctx.providers_attempted, normalized_name)
        for name in status.get("skipped_providers") or []:
            if str(name):
                _append_unique(ctx.providers_skipped, str(name))
        ctx.provider_failure_reasons.update(
            {
                str(key): str(value)
                for key, value in dict(status.get("provider_failure_reasons") or {}).items()
            }
        )
        if provider and provider not in {"cache", "none", "indexed_external"}:
            ctx.search_providers.add(provider)

        intent = _retrieval_intent(self._intent(ctx), relevance_query)
        candidates: list[ExternalCandidate] = []
        hit_by_candidate: dict[str, object] = {}
        excluded = {str(value) for value in excluded_provenance_roots if str(value)}
        for hit in hits:
            candidate = candidate_from_search_hit(
                hit,
                plan_id=plan_id,
                query=query,
                intent=intent,
            )
            if candidate.provenance_root and candidate.provenance_root in excluded:
                ctx.retrieval_rejection_reason_codes.add("excluded_provenance_root")
                continue
            candidates.append(candidate)
            hit_by_candidate[candidate.candidate_id] = hit
            if candidate.prefetch_decision == "rejected":
                ctx.retrieval_rejection_reason_codes.update(candidate.reason_codes or ("relevance_rejected",))

        candidates.sort(
            key=lambda item: (
                {"candidate": 0, "unknown": 1, "rejected": 2}.get(item.prefetch_decision, 3),
                {"satisfied": 0, "not_requested": 1, "uncertain": 2, "unsatisfied": 3}.get(
                    item.authority_status,
                    2,
                ),
                {"strong": 0, "weak": 1, "uncertain": 2, "unrelated": 3}.get(
                    item.relevance,
                    2,
                ),
                -title_intent_fit_score(item),
                -title_query_phrase_span(item),
                -len(item.matched_anchors),
                item.rank,
                item.provenance_root,
                item.candidate_id,
            )
        )
        ctx.retrieval_budget = replace(
            retrieval,
            candidate_count=retrieval.candidate_count + len(candidates),
        )

        evidence: list[EvidenceItem] = []
        indexed_document_ids: list[str] = []
        for candidate in candidates:
            hit = hit_by_candidate[candidate.candidate_id]
            metadata = getattr(hit, "metadata", {})
            metadata = dict(metadata) if isinstance(metadata, dict) else {}
            if tier == "indexed_external" and metadata.get("document_id"):
                indexed_document_ids.append(str(metadata["document_id"]))
                continue
            item = snippet_evidence(candidate)
            if item is not None:
                evidence.append(item)

        if indexed_document_ids:
            evidence.extend(
                self._retriever.retrieve(
                    query,
                    top_k=budget.top_k,
                    excluded_provenance_roots=excluded_provenance_roots,
                    document_ids=tuple(dict.fromkeys(indexed_document_ids)),
                )
            )

        readiness_target = 2 if request.mode is Mode.MULTIPLE else 1
        ready_count = max(0, int(prior_ready_count)) + len(_dedup_evidence(evidence))
        _LOG.info(
            "external_candidates_assessed",
            run_id=rid,
            plan_id=plan_id,
            candidates=len(candidates),
            candidate_ready=sum(1 for item in candidates if item.prefetch_decision == "candidate"),
            candidate_unknown=sum(1 for item in candidates if item.prefetch_decision == "unknown"),
            candidate_rejected=sum(1 for item in candidates if item.prefetch_decision == "rejected"),
            prior_ready=max(0, int(prior_ready_count)),
            ready_count=ready_count,
            readiness_target=readiness_target,
        )
        if ready_count < readiness_target:
            fetch_pool = [item for item in candidates if item.prefetch_decision == "unknown"]
            fetched = self._fetch_external_candidates(
                fetch_pool,
                intent=intent,
                request=request,
                plan_id=plan_id,
                rid=rid,
                ctx=ctx,
                recovery=recovery,
                deadline=deadline,
                readiness_needed=max(0, readiness_target - ready_count),
            )
            evidence.extend(fetched)

        self._emit_search_degraded(rid, sink, status=status)
        status["providers_attempted"] = list(ctx.providers_attempted)
        status["skipped_providers"] = list(ctx.providers_skipped)
        status["provider_failure_reasons"] = dict(ctx.provider_failure_reasons)
        return {
            "evidence": _dedup_evidence(evidence)[: budget.top_k],
            "status": status,
            "provider": provider or None,
            "tier": tier or ("indexed_external" if evidence else None),
            "locale": status.get("locale") if isinstance(status.get("locale"), dict) else None,
            "reason": "provider_degraded" if status.get("live_providers_down") else None,
        }

    def _search_external_candidates(
        self,
        query: str,
        *,
        count: int,
        options: SearchOptions | None,
        request: RunRequest,
        plan_id: str,
        recovery: bool,
        excluded_provenance_roots: tuple[str, ...],
        relevance_query: str | None,
        deadline: _RunDeadline | None,
        ctx: _RunContext,
        rid: str,
        search_handle: _ExternalSearchHandle | None,
    ) -> tuple[list[object], dict[str, object]]:
        """Apply the live/cache/indexed policy with Store access on the caller thread."""

        search = self._search
        if search is None:
            return [], {}
        metric_started_at = (
            search_handle.started_at
            if search_handle is not None and search_handle.started_at > 0
            else time.time()
        )
        metric_remaining_before = (
            search_handle.remaining_before_s if search_handle is not None else None
        )
        temporal_intent, _ = detect_temporal_intent(request.query)
        provider_relevance_query = relevance_query or self._intent(ctx).original_query or request.query
        relevance_kwargs = _search_relevance_kwargs(search, provider_relevance_query)
        read_cache = getattr(search, "read_search_cache", None)
        search_network = getattr(search, "search_network", None)
        persist_cache = getattr(search, "persist_search_cache", None)
        read_indexed = getattr(search, "read_indexed_external", None)
        result: SearchCallResult | None = search_handle.result if search_handle is not None else None
        usage_kwargs = _search_usage_kwargs(search, request)

        # Stable warm runs may start from the fresh locale-v4 cache. Latest/current
        # and recovery runs must attempt live first.
        if result is None and temporal_intent == "none" and not recovery and not ctx.fresh_search_required and callable(read_cache):
            result = read_cache(
                query,
                count=count,
                options=options,
                **relevance_kwargs,
                **usage_kwargs,
            )

        if result is None and search_handle is not None and search_handle.future is not None:
            timeout = None
            if deadline is not None:
                retrieval = ctx.retrieval_budget or retrieval_budget_for(request.mode)
            try:
                result = self._await_network_future(
                    search_handle.future,
                    run_id=rid,
                    deadline=deadline,
                    reserve_s=(
                        (ctx.retrieval_budget or retrieval_budget_for(request.mode)).reserve_total_s
                        if deadline is not None
                        else 0.0
                    ),
                )
            except FutureTimeoutError as error:
                if deadline is not None:
                    self._append_phase_metric(
                        ctx,
                        stage="candidate_search",
                        plan_id=plan_id,
                        lane="external",
                        query=query,
                        started_at=metric_started_at,
                        finished_at=time.time(),
                        status="timeout",
                        deadline=deadline,
                        remaining_before_s=metric_remaining_before,
                    )
                raise BaselineError(
                    ErrorCode.RUN_TIMEOUT,
                    "Run deadline was exceeded during external candidate search.",
                ) from error
            except BaselineError as error:
                if deadline is not None:
                    self._append_phase_metric(
                        ctx,
                        stage="candidate_search",
                        plan_id=plan_id,
                        lane="external",
                        query=query,
                        started_at=metric_started_at,
                        finished_at=time.time(),
                        status=("timeout" if error.code is ErrorCode.RUN_TIMEOUT else "failed"),
                        deadline=deadline,
                        remaining_before_s=metric_remaining_before,
                    )
                raise
            finally:
                if search_handle.future.done() and not search_handle.future.cancelled():
                    ctx.provider_circuit_open.update(search_handle.provider_circuit)
                self._close_external_search_handle(search_handle)
            if deadline is not None:
                self._check_deadline(
                    deadline,
                    reserve_s=(ctx.retrieval_budget or retrieval_budget_for(request.mode)).reserve_total_s,
                )

        if result is None and callable(search_network):
            if deadline is not None:
                retrieval = ctx.retrieval_budget or retrieval_budget_for(request.mode)
                self._check_deadline(deadline, reserve_s=retrieval.reserve_total_s)
            result = search_network(
                query,
                count=count,
                options=options,
                recovery=recovery or ctx.fresh_search_required,
                excluded_provenance_roots=excluded_provenance_roots,
                relevance_query=provider_relevance_query,
                **_provider_circuit_kwargs(search, ctx),
                **usage_kwargs,
            )
            if deadline is not None:
                self._check_deadline(
                    deadline,
                    reserve_s=(ctx.retrieval_budget or retrieval_budget_for(request.mode)).reserve_total_s,
                )
        if (
            result is not None
            and result.hits
            and callable(persist_cache)
            and str(result.status.get("producer_tier") or "")
            in {"brave", "tavily", "exa", "firecrawl", "firecrawl_search"}
        ):
            if deadline is not None:
                self._check_deadline(
                    deadline,
                    reserve_s=(ctx.retrieval_budget or retrieval_budget_for(request.mode)).reserve_total_s,
                )
            persist_cache(query, count=count, options=options, result=result, **usage_kwargs)

        upstream_status = dict(result.status) if result is not None else {}
        if (result is None or not result.hits) and not recovery and not ctx.fresh_search_required and callable(read_cache):
            cached = read_cache(
                query,
                count=count,
                options=options,
                **relevance_kwargs,
                upstream_status=upstream_status,
                **usage_kwargs,
            )
            if cached is not None:
                result = cached

        if (result is None or not result.hits) and callable(read_indexed):
            result = read_indexed(
                query,
                count=count,
                options=options,
                recovery=recovery or ctx.fresh_search_required,
                excluded_provenance_roots=excluded_provenance_roots,
                relevance_query=provider_relevance_query,
                upstream_status=upstream_status,
                **usage_kwargs,
            )

        if result is None:
            try:
                raw_hits = search.search(query, count=count, options=options, **usage_kwargs)
            except TypeError:
                raw_hits = search.search(query, count=count)
            status = self._search_status()
            if deadline is not None:
                self._append_phase_metric(
                    ctx,
                    stage="candidate_search",
                    plan_id=plan_id,
                    lane="external",
                    query=query,
                    provider=str(status.get("producer_provider") or status.get("fallback_tier") or "") or None,
                    started_at=metric_started_at,
                    finished_at=time.time(),
                    status="completed" if raw_hits else "empty",
                    deadline=deadline,
                    remaining_before_s=metric_remaining_before,
                )
            return list(raw_hits or ()), status
        if deadline is not None:
            self._append_phase_metric(
                ctx,
                stage="candidate_search",
                plan_id=plan_id,
                lane="external",
                query=query,
                provider=str(result.status.get("producer_provider") or result.status.get("producer_tier") or "") or None,
                started_at=metric_started_at,
                finished_at=time.time(),
                status="completed" if result.hits else "empty",
                deadline=deadline,
                remaining_before_s=metric_remaining_before,
            )
        return list(result.hits), dict(result.status)

    def _fetch_external_candidates(
        self,
        candidates: list[ExternalCandidate],
        *,
        intent: ResearchIntent,
        request: RunRequest,
        plan_id: str,
        rid: str,
        ctx: _RunContext,
        recovery: bool,
        deadline: _RunDeadline | None,
        readiness_needed: int,
    ) -> list[EvidenceItem]:
        """Fetch a bounded deterministic candidate prefix and commit on this thread."""

        retrieval = ctx.retrieval_budget or retrieval_budget_for(request.mode)
        if recovery:
            remaining = max(0, retrieval.recovery_fetches_max - retrieval.recovery_fetches_used)
        else:
            remaining = max(0, retrieval.fetches_max - retrieval.fetches_used)
        limit = min(len(candidates), remaining, max(1, readiness_needed))
        if readiness_needed <= 0 or limit <= 0 or self._search is None or self._indexer is None:
            return []
        selected = candidates[:limit]
        fetch_started_at = {candidate.candidate_id: time.time() for candidate in selected}
        documents: dict[str, Document] = {}
        network_selected: list[ExternalCandidate] = []
        read_cache = getattr(self._search, "read_fetch_cache", None)
        usage_kwargs = _search_usage_kwargs(self._search, request)
        focused_fetch_kwargs = _focused_fetch_kwargs(
            self._search,
            intent.research_focus,
        )
        temporal_intent, _ = detect_temporal_intent(request.query)

        for candidate in selected:
            cached = None
            if temporal_intent == "none" and not recovery and not ctx.fresh_search_required and callable(read_cache):
                cached = read_cache(
                    candidate.url,
                    **focused_fetch_kwargs,
                    **usage_kwargs,
                )
            if cached is not None:
                documents[candidate.candidate_id] = cached
            else:
                network_selected.append(candidate)

        fetch_network = getattr(self._search, "fetch_network", None)
        fetch_budget = (
            deadline.provider_budget(
                phase_reserve_s=retrieval.reserve_total_s,
                retrieval_budget=retrieval,
                interrupt_check=self._network_interrupt_checker(rid, deadline),
                max_attempts=2,
            )
            if deadline is not None and network_selected
            else None
        )
        if network_selected and callable(fetch_network) and retrieval.fetch_concurrency > 1:

            def _fetch_one(candidate: ExternalCandidate) -> Document:
                with use_provider_call_budget(fetch_budget):
                    return fetch_network(
                        candidate.url,
                        **_provider_circuit_kwargs(self._search, ctx),
                        **focused_fetch_kwargs,
                        **usage_kwargs,
                    )

            executor = ThreadPoolExecutor(
                max_workers=min(retrieval.fetch_concurrency, len(network_selected)),
                thread_name_prefix="t49-fetch",
            )
            futures: dict[str, Future] = {
                candidate.candidate_id: executor.submit(_fetch_one, candidate)
                for candidate in network_selected
            }
            try:
                for candidate in network_selected:
                    documents[candidate.candidate_id] = self._await_network_future(
                        futures[candidate.candidate_id],
                        run_id=rid,
                        deadline=deadline,
                        reserve_s=retrieval.reserve_total_s,
                    )
            except FutureTimeoutError as error:
                for future in futures.values():
                    future.cancel()
                if deadline is not None:
                    for candidate in network_selected:
                        self._append_phase_metric(
                            ctx,
                            stage="document_fetch",
                            plan_id=plan_id,
                            lane="external",
                            query=candidate.query,
                            started_at=fetch_started_at[candidate.candidate_id],
                            finished_at=time.time(),
                            status="timeout",
                            deadline=deadline,
                            candidate_id=candidate.candidate_id,
                        )
                raise BaselineError(
                    ErrorCode.RUN_TIMEOUT,
                    "Run deadline was exceeded during external document fetch.",
                ) from error
            except BaselineError as error:
                if deadline is not None:
                    for candidate in network_selected:
                        self._append_phase_metric(
                            ctx,
                            stage="document_fetch",
                            plan_id=plan_id,
                            lane="external",
                            query=candidate.query,
                            started_at=fetch_started_at[candidate.candidate_id],
                            finished_at=time.time(),
                            status=("timeout" if error.code is ErrorCode.RUN_TIMEOUT else "failed"),
                            deadline=deadline,
                            candidate_id=candidate.candidate_id,
                        )
                raise
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        else:
            for candidate in network_selected:
                if deadline is not None:
                    self._check_deadline(deadline, reserve_s=retrieval.reserve_total_s)
                with use_provider_call_budget(fetch_budget):
                    if callable(fetch_network):
                        document = fetch_network(
                            candidate.url,
                            **_provider_circuit_kwargs(self._search, ctx),
                            **focused_fetch_kwargs,
                            **usage_kwargs,
                        )
                    else:
                        document = self._search.fetch(
                            candidate.url,
                            **_provider_circuit_kwargs(self._search, ctx),
                            **focused_fetch_kwargs,
                            **usage_kwargs,
                        )
                if deadline is not None:
                    self._check_deadline(deadline, reserve_s=retrieval.reserve_total_s)
                documents[candidate.candidate_id] = document

        used = len(selected)
        if recovery:
            ctx.retrieval_budget = replace(
                ctx.retrieval_budget or retrieval,
                recovery_fetches_used=retrieval.recovery_fetches_used + used,
            )
        else:
            ctx.retrieval_budget = replace(
                ctx.retrieval_budget or retrieval,
                fetches_used=retrieval.fetches_used + used,
            )

        persist_cache = getattr(self._search, "persist_fetch_cache", None)
        evidence: list[EvidenceItem] = []
        for candidate in selected:
            document = documents.get(candidate.candidate_id)
            if document is None:
                continue
            for name in document.metadata.get("attempted_extract_providers") or []:
                if str(name):
                    _append_unique(ctx.providers_attempted, str(name))
            for name in document.metadata.get("skipped_extract_providers") or []:
                if str(name):
                    _append_unique(ctx.providers_skipped, str(name))
            ctx.provider_failure_reasons.update(
                {
                    str(key): str(value)
                    for key, value in dict(document.metadata.get("provider_failure_reasons") or {}).items()
                }
            )
            canonical_url = str(document.metadata.get("canonical_url") or "")
            fetched = bind_fetched_content(
                candidate,
                intent=intent,
                document_url=document.url,
                title=document.title,
                text=document.text,
                canonical_url=canonical_url or None,
                fetch_provider=str(document.metadata.get("fetch_provider") or ""),
            )
            if deadline is not None:
                self._append_phase_metric(
                    ctx,
                    stage="document_fetch",
                    plan_id=plan_id,
                    lane="external",
                    query=candidate.query,
                    provider=str(document.metadata.get("fetch_provider") or "") or None,
                    started_at=fetch_started_at[candidate.candidate_id],
                    finished_at=time.time(),
                    status=("rejected" if fetched.prefetch_decision == "rejected" else "completed"),
                    deadline=deadline,
                    candidate_id=candidate.candidate_id,
                )
            if not fetched_body_may_contain_focused_evidence(fetched):
                ctx.retrieval_rejection_reason_codes.update(
                    fetched.reason_codes or ("fetched_evidence_rejected",)
                )
                if candidate.provenance_root:
                    ctx.unready_external_provenance_roots.add(candidate.provenance_root)
                continue
            if deadline is not None:
                self._check_deadline(deadline, reserve_s=retrieval.reserve_total_s)
            prepare_document = getattr(self._indexer, "prepare_document", None)
            if callable(prepare_document):
                prepare_started_at = time.time()
                prepared = self._prepare_transient_external_document(
                    prepare_document,
                    document,
                )
                self._check_external_write_allowed(
                    rid,
                    deadline,
                    reserve_s=retrieval.reserve_total_s,
                )
                sanitized_text = str(
                    getattr(getattr(prepared, "sanitized", None), "text", "")
                    or document.text
                )
                prepared_pieces = tuple(getattr(prepared, "pieces", ()) or ())
                excerpt, selected_piece_index = select_fetched_evidence_excerpt(
                    intent,
                    prepared_pieces or (sanitized_text,),
                    document_url=canonical_url or document.url,
                    document_title=document.title,
                )
                fetched = bind_fetched_content(
                    candidate,
                    intent=intent,
                    document_url=document.url,
                    title=document.title,
                    text=excerpt,
                    canonical_url=canonical_url or None,
                    fetch_provider=str(document.metadata.get("fetch_provider") or ""),
                )
                if fetched.prefetch_decision != "candidate" or not fetched.snippet.strip():
                    ctx.retrieval_rejection_reason_codes.update(
                        fetched.reason_codes or ("prepared_evidence_rejected",)
                    )
                    if candidate.provenance_root:
                        ctx.unready_external_provenance_roots.add(candidate.provenance_root)
                    if callable(persist_cache):
                        persist_cache(
                            document,
                            **focused_fetch_kwargs,
                            **usage_kwargs,
                        )
                    continue
                if deadline is not None:
                    self._append_phase_metric(
                        ctx,
                        stage="document_prepare",
                        plan_id=plan_id,
                        lane="external",
                        query=candidate.query,
                        provider=(
                            str(
                                getattr(
                                    getattr(self._indexer, "_embedder", None),
                                    "name",
                                    "",
                                )
                                or ""
                            )
                            or None
                            if getattr(prepared, "embeddings", None)
                            else None
                        ),
                        started_at=prepare_started_at,
                        finished_at=time.time(),
                        status="completed",
                        deadline=deadline,
                        candidate_id=candidate.candidate_id,
                    )
                    self._check_deadline(deadline, reserve_s=retrieval.reserve_total_s)
                if fetched.provenance_root:
                    ctx.prepared_external_documents[fetched.provenance_root] = (
                        prepared,
                        selected_piece_index,
                    )
                if callable(persist_cache):
                    persist_cache(
                        document,
                        **focused_fetch_kwargs,
                        **usage_kwargs,
                    )
            else:
                self._check_external_write_allowed(
                    rid,
                    deadline,
                    reserve_s=retrieval.reserve_total_s,
                )
                if callable(persist_cache):
                    persist_cache(
                        document,
                        **focused_fetch_kwargs,
                        **usage_kwargs,
                    )
            evidence.append(
                EvidenceItem(
                    chunk_id=-1,
                    document_id=fetched.provenance_root,
                    url=fetched.canonical_url or fetched.url,
                    title=fetched.title,
                    text=fetched.snippet,
                    score=1.0 / (1 + fetched.rank),
                    channels=["fetched"],
                    provenance_refs=[fetched.provenance_root] if fetched.provenance_root else [],
                    published_at=fetched.published_at,
                    updated_at=fetched.updated_at,
                    effective_at=fetched.effective_at,
                    retrieved_at=fetched.retrieved_at or time.time(),
                    search_provider=fetched.provider,
                    fetch_provider=fetched.fetch_provider,
                )
            )
            if len(evidence) >= readiness_needed:
                break
        return evidence

    def _temporal_assurance(
        self,
        request: RunRequest,
        budget: Budget,
        rid: str,
        sink: EventSink | None,
        ctx: _RunContext,
        evidence: list[EvidenceItem],
        *,
        plan_id: str,
        deadline: _RunDeadline | None = None,
    ) -> tuple[list[EvidenceItem], TemporalCoverage]:
        """Run one extra three-lane assurance query when temporal intent requires it."""

        intent, requested_as_of = detect_temporal_intent(request.query)
        assurance_evidence: list[EvidenceItem] = []
        if intent != "none":
            assurance_query = temporal_validation_query(
                request.query,
                intent=intent,
                requested_as_of=requested_as_of,
            )
            assurance_evidence = self._retrieve_evidence_plan(
                assurance_query,
                request,
                budget,
                rid,
                sink,
                ctx,
                plan_id=plan_id,
                deadline=deadline,
            )
        combined = _fuse_evidence_rankings(
            [ranking for ranking in (evidence, assurance_evidence) if ranking],
            budget.top_k,
        )
        coverage = assess_temporal_coverage(
            intent=intent,
            requested_as_of=requested_as_of,
            evidence=combined,
            checked_at=time.time(),
        )
        self._replace_temporal_reasons(ctx, coverage)
        return combined, coverage

    @staticmethod
    def _ensure_citable_evidence(evidence: list[EvidenceItem]) -> None:
        if any(item.grounding_eligible and item.text.strip() for item in evidence):
            return
        raise BaselineError(
            ErrorCode.GROUNDING_REQUIRED,
            "No citable evidence was retrieved for this research query.",
            details={"reason": "zero_citable_evidence", "evidence_count": 0},
        )

    @staticmethod
    def _add_degraded_reason(ctx: _RunContext, reason: str) -> None:
        if reason and reason not in ctx.degraded_reasons:
            ctx.degraded_reasons.append(reason)

    def _search_status(self) -> dict:
        status_fn = getattr(self._search, "last_search_status", None)
        if not callable(status_fn):
            return {}
        status = status_fn()
        return dict(status) if isinstance(status, dict) else {}

    def _ingest(
        self,
        query: str,
        budget: Budget,
        rid: str,
        sink: EventSink | None = None,
        *,
        options: SearchOptions | None = None,
        provenance: set[str] | None = None,
        recovery: bool = False,
        excluded_provenance_roots: tuple[str, ...] = (),
        relevance_query: str | None = None,
    ) -> list[dict]:
        stats, _ = self._ingest_with_status(
            query,
            budget,
            rid,
            sink,
            options=options,
            provenance=provenance,
            recovery=recovery,
            excluded_provenance_roots=excluded_provenance_roots,
            relevance_query=relevance_query,
        )
        return stats

    def _ingest_with_status(
        self,
        query: str,
        budget: Budget,
        rid: str,
        sink: EventSink | None = None,
        *,
        options: SearchOptions | None = None,
        provenance: set[str] | None = None,
        recovery: bool = False,
        excluded_provenance_roots: tuple[str, ...] = (),
        relevance_query: str | None = None,
    ) -> tuple[list[dict], dict]:
        if not self._search or not self._indexer:
            return [], {}
        try:
            index_with_outcome = getattr(self._indexer, "index_search_results_with_outcome", None)
            if callable(index_with_outcome):
                kwargs = {
                    "count": max(1, budget.max_searches),
                    "options": options,
                    "recovery": recovery,
                    "excluded_provenance_roots": excluded_provenance_roots,
                    "relevance_query": relevance_query,
                }
                if "adoption_run_id" in inspect.signature(index_with_outcome).parameters:
                    kwargs["adoption_run_id"] = rid
                stats, status = index_with_outcome(self._search, query, **kwargs)
            else:
                kwargs = {"count": max(1, budget.max_searches), "options": options}
                index_search = self._indexer.index_search_results
                if "adoption_run_id" in inspect.signature(index_search).parameters:
                    kwargs["adoption_run_id"] = rid
                stats = index_search(self._search, query, **kwargs)
                status = self._search_status()
            if provenance is not None:
                provenance.update(_stats_search_providers(stats))
            self._emit_search_degraded(rid, sink, status=status)
            return stats, status
        except BaselineError:
            raise
        except Exception:  # noqa: BLE001 - retrieval can still use prior index
            status = self._search_status()
            self._emit_search_degraded(rid, sink, status=status)
            return [], status

    @staticmethod
    def _execution_provenance(ctx: _RunContext) -> dict[str, object]:
        """Return only adapter- and indexer-observed execution identities."""

        return {
            "schema": "deepresearch.execution-provenance/v1",
            "model_observation": "resolved_adapter",
            "search_observation": "indexer_stats",
            "llm_model_refs": list(ctx.llm_model_refs),
            "search_providers": sorted(ctx.search_providers),
        }

    def _emit_search_degraded(
        self,
        rid: str,
        sink: EventSink | None,
        *,
        status: dict | None = None,
    ) -> None:
        status = status if status is not None else self._search_status()
        if not isinstance(status, dict) or not status.get("live_providers_down"):
            return
        tier = str(status.get("fallback_tier") or "none")
        self._emit(
            rid,
            "search_degraded",
            {"code": ErrorCode.SEARCH_PROVIDER_DOWN.value, "fallback_tier": tier},
            sink,
        )

    def _check_deadline(self, deadline: _RunDeadline, *, reserve_s: float = 0.0) -> None:
        if deadline.expired(reserve_s=reserve_s):
            self._raise_timeout()

    def _check_external_write_allowed(
        self,
        run_id: str,
        deadline: _RunDeadline | None,
        *,
        reserve_s: float,
    ) -> None:
        """Reject cache/index promotion after cancellation, deadline, or terminal state."""

        self._check_cancel(run_id)
        if deadline is not None:
            self._check_deadline(deadline, reserve_s=reserve_s)
        run = self._store.get_run(run_id) or {}
        if str(run.get("status") or "") != "running":
            raise BaselineError(
                ErrorCode.RUN_CANCELLED,
                "run reached a terminal state before external evidence was persisted",
            )

    @staticmethod
    def _raise_timeout() -> None:
        raise BaselineError(ErrorCode.RUN_TIMEOUT, "Run timed out before evidence was available.")

    def _finalize(
        self,
        answer: GroundedAnswer,
        mode: Mode,
        rid: str,
        evidence: list[EvidenceItem],
        *,
        ctx: _RunContext | None = None,
        sink: EventSink | None = None,
        conflict_analysis: tuple[list, list[dict]] | None = None,
        deadline: _RunDeadline | None = None,
    ) -> None:
        if deadline is not None:
            retrieval = ctx.retrieval_budget if ctx is not None else retrieval_budget_for(mode)
            self._check_cancel(rid)
            self._check_deadline(deadline, reserve_s=retrieval.cleanup_reserve_s)
            current = self._store.get_run(rid) or {}
            if str(current.get("status") or "") != "running":
                raise BaselineError(
                    ErrorCode.RUN_CANCELLED,
                    "run reached a terminal state before citations were finalized",
                )
        # The report is committed independently; verification updates metadata later.
        answer.confidence_label = ""
        if ctx is not None:
            ctx.final_adopted_evidence = list(evidence)

    def _start_verification(self, request: RunRequest, evidence: list[EvidenceItem], ctx: _RunContext) -> None:
        """Check evidence in parallel with writing, without Store access in the thread."""
        if ctx.verification_future is not None:
            return
        ctx.verification_started_at = time.time()
        ctx.verification_deadline = time.monotonic() + 20.0
        snapshot = [replace(item, channels=list(item.channels), provenance_refs=list(item.provenance_refs)) for item in evidence]

        def verify_evidence() -> tuple[list, list[dict]]:
            with use_provider_call_budget(ProviderCallBudget(
                request_deadline=ctx.verification_deadline, max_attempts=1,
            )):
                return analyze_evidence_pack_conflicts(snapshot, llm=ctx.llm, query=request.query)

        ctx.verification_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="report-verification")
        ctx.verification_future = ctx.verification_executor.submit(verify_evidence)

    def _complete_verification(self, request: RunRequest, outcome: RunOutcome, ctx: _RunContext) -> None:
        """Persist advisory results after report delivery; never retract report content."""
        reasons = list(ctx.verification_reasons)
        status = "completed"
        conflicts: list = []
        decisions: list[dict] = []
        try:
            enforce_grounding(outcome.answer, min_coverage=_MIN_COVERAGE.get(request.mode, 0.85))
        except BaselineError as error:
            reasons.append(str(error.details.get("reason") or "unsupported_claim"))
        except Exception:
            status = "failed"
            reasons.append("verification_incomplete")
        try:
            if ctx.verification_future is not None:
                conflicts, decisions = ctx.verification_future.result(
                    timeout=max(0.0, ctx.verification_deadline - time.monotonic()),
                )
                if any(item.get("basis") == "classification_unavailable" for item in decisions):
                    reasons.append("verification_incomplete")
            if conflicts:
                reasons.append("source_conflict")
        except Exception:
            status = "failed"
            reasons.append("verification_incomplete")
        reasons = list(dict.fromkeys(reasons))
        agreement = outcome.extra.get("breakdown", {}).get("agreement") if request.mode is Mode.COUNCIL else None
        label = "uncertain" if reasons else derive_confidence_label(outcome.answer.coverage, agreement=agreement)
        if label == "uncertain" and not reasons:
            reasons.append("evidence_insufficient")
        verification = {
            "status": status, "started_at": ctx.verification_started_at or time.time(),
            "completed_at": time.time(), "coverage": outcome.answer.coverage,
            "conflict_decisions": [dict(decision) for decision in decisions],
        }
        warning = {"reasons": reasons, "acknowledged": False} if reasons else None
        try:
            saved = self._store.update_run_verification(
                outcome.run_id, verification=verification, confidence_label=label,
                confidence_warning=warning, conflicts=[item.to_dict() for item in conflicts],
            )
            if saved:
                outcome.answer.confidence_label = label
                outcome.answer.conflicts = list(conflicts)
                outcome.extra["verification"] = verification
                if warning:
                    outcome.extra["confidence_warning"] = warning
        except Exception as error:
            # A later GET detects interrupted verification; the delivered report stays intact.
            _LOG.warning("report_verification_persist_failed", run_id=outcome.run_id, error_type=type(error).__name__)

    def _promote_final_external_evidence(self, evidence: list[EvidenceItem]) -> None:
        """Best-effort expose only grounded final external evidence in Library."""

        promote = getattr(self._store, "promote_document_to_library", None)
        if not callable(promote):
            return
        for item in evidence:
            if item.origin == "external" and str(item.url or "").lower().startswith(("http://", "https://")):
                try:
                    promote(item.document_id)
                except Exception:  # noqa: BLE001 - Library projection must not invalidate a grounded answer
                    continue

    @staticmethod
    def _citation_rows(citations: list[Citation]) -> list[dict[str, object]]:
        """Return Store rows for an atomic run-terminal transition."""

        return [
            {
                "document_id": citation.document_id,
                "url": citation.url,
                "title": citation.title,
                "snippet": citation.snippet,
                "score": citation.score,
                "origin": citation.origin,
                "artifact_type": citation.artifact_type,
                "artifact_id": citation.artifact_id,
                "grounding_eligible": citation.grounding_eligible,
                "provenance_refs": list(citation.provenance_refs),
                "source_status": citation.source_status,
                "published_at": citation.published_at,
                "updated_at": citation.updated_at,
                "effective_at": citation.effective_at,
                "retrieved_at": citation.retrieved_at,
            }
            for citation in citations
        ]

    def _persist_citations(self, run_id: str, citations: list[Citation]) -> None:
        """Compatibility helper for direct persistence callers.

        Normal run finalization persists citations atomically with the terminal
        state through ``Store.transition_run``.  Keep this narrow helper for
        migrations and tests that exercise citation serialization in isolation.
        """

        for row in self._citation_rows(citations):
            self._store.add_citation(run_id=run_id, **row)

    def _merge_lanes(self, rankings: list[list[int]], top_k: int) -> list[EvidenceItem]:
        fused = reciprocal_rank_fusion(rankings)
        ordered = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        items: list[EvidenceItem] = []
        for chunk_id, score in ordered[: top_k * 2]:
            chunk = self._store.get_chunk(chunk_id)
            if not chunk:
                continue
            doc = self._store.get_document(chunk.document_id) or {}
            items.append(
                EvidenceItem(
                    chunk_id=chunk.id, document_id=chunk.document_id,
                    url=doc.get("url", ""), title=doc.get("title", ""), text=chunk.text, score=score,
                )
            )
        return _dedup_evidence(items)[:top_k]

    @staticmethod
    def _merge_evidence_rankings(
        rankings: list[list[EvidenceItem]], top_k: int
    ) -> list[EvidenceItem]:
        """Merge query results without treating repeat recall as corroboration."""

        return _fuse_evidence_rankings(
            rankings,
            top_k,
            preserve_ranking_diversity=True,
        )

    def _subqueries(
        self,
        query: str,
        count: int,
        ctx: _RunContext,
        *,
        options: SearchOptions | None = None,
    ) -> list[str]:
        if count <= 1:
            return [query]
        subs: list[str] = []
        try:
            result = ctx.llm.complete(
                [{"role": "user", "content": _SUBQUERY_PROMPT.format(query=query, n=count)}],
                temperature=0.3, max_tokens=256,
            )
            lines = [line.strip("-* 	").strip() for line in result.text.splitlines() if line.strip()]
            subs = [line for line in lines if len(line) > 3]
        except Exception:  # noqa: BLE001 - fall back to the original query
            pass
        if not subs:
            subs = [query]
        return _merge_extra_subqueries([query, *subs], count=count, options=options)

    def _plan(
        self,
        query: str,
        budget: Budget,
        ctx: _RunContext,
        *,
        section_count: int | None = None,
    ) -> list[str]:
        requested_count = (
            max(1, min(int(budget.max_searches), int(section_count)))
            if section_count
            else None
        )
        candidate_limit = max(5, requested_count or 0)
        prompt = (
            _PLAN_COUNT_PROMPT.format(query=query, n=requested_count)
            if requested_count is not None
            else _PLAN_PROMPT.format(query=query)
        )
        try:
            result = ctx.llm.complete(
                [{"role": "user", "content": prompt}],
                temperature=0.3, max_tokens=budget.max_think_tokens,
            )
            sections = [line.strip("-* 	0123456789.").strip() for line in result.text.splitlines() if line.strip()]
            sections = [
                s
                for s in sections
                if len(s) > 4 and s.lower() not in _GAP_SENTINELS
            ][:candidate_limit]
            if sections:
                return (
                    _select_deep_sections(sections, query, requested_count)
                    if requested_count is not None
                    else sections
                )
        except Exception:  # noqa: BLE001
            pass
        return [query]

    def _maybe_remember(self, request: RunRequest, outcome: RunOutcome) -> str | None:
        if not self._memory or not request.session_id:
            return None
        summary = outcome.answer.text[:500]
        if summary:
            # The terminal event is already public here. Optional remote embeddings
            # must not keep the only research worker busy and consume the next
            # queued run's absolute deadline.
            return self._memory.remember(
                MemoryLayer.SESSION,
                request.session_id,
                summary,
                salience=1.0,
                embed=False,
            )
        return None

    def _persist_messages(self, request: RunRequest, outcome: RunOutcome, rid: str) -> list[str]:
        """Record the user query and assistant answer as session messages."""

        if not request.session_id:
            return []
        # Persist recent-turn continuity synchronously but skip optional remote
        # embeddings after terminal publication. Semantic enrichment is optional
        # and tracked separately; it cannot delay the next research job.
        message_ids = [
            self._history.index_turn(
                request.session_id,
                "user",
                request.query,
                metadata={"run_id": rid},
                embed=False,
            )
        ]
        answer_text = outcome.answer.text
        if answer_text:
            message_ids.append(
                self._history.index_turn(
                    request.session_id,
                    "assistant",
                    answer_text,
                    metadata={"run_id": rid, "mode": request.mode.value},
                    embed=False,
                )
            )
        return message_ids

    def _emit(self, run_id: str, event_type: str, payload: dict, sink: EventSink | None) -> None:
        seq = self._store.append_event(run_id, event_type, payload)
        if sink:
            sink({"seq": seq, "type": event_type, "payload": payload})

    def _emit_budget(
        self,
        run_id: str,
        sink: EventSink | None,
        *,
        mode: Mode,
        used: int,
        maximum: int,
        stage: str,
        gap_rounds: int | None = None,
        pages_fetched: int | None = None,
        remaining_searches: int | None = None,
        search_tier: str | None = None,
        degraded_fetch: bool | None = None,
        recovery_searches_max: int | None = None,
        recovery_searches_used: int | None = None,
    ) -> None:
        """Emit a ``budget_snapshot`` so the UI can show searches used vs. budget."""

        payload: dict = {
            "mode": mode.value,
            "searches_used": used,
            "searches_max": maximum,
            "stage": stage,
        }
        if gap_rounds is not None:
            payload["gap_rounds"] = gap_rounds
        if pages_fetched is not None:
            payload["pages_fetched"] = pages_fetched
        if remaining_searches is not None:
            payload["remaining_searches"] = remaining_searches
        if search_tier:
            payload["search_tier"] = search_tier
        if degraded_fetch is not None:
            payload["degraded_fetch"] = degraded_fetch
        if recovery_searches_max is not None:
            payload["recovery_searches_max"] = recovery_searches_max
        if recovery_searches_used is not None:
            payload["recovery_searches_used"] = recovery_searches_used
        self._emit(run_id, "budget_snapshot", payload, sink)

    def _check_cancel(self, run_id: str) -> None:
        with self._lock:
            cancelled = run_id in self._cancelled
        if not cancelled:
            # Honor a cancel requested via a different (rewired) orchestrator instance,
            # which only mutates the persisted run status.
            run = self._store.get_run(run_id)
            cancelled = bool(run and run.get("status") == "cancelling")
        if cancelled:
            raise BaselineError(ErrorCode.RUN_CANCELLED, "run was cancelled")


def _env_int(name: str, default: int) -> int:
    """Read a positive int from env, falling back to ``default`` on missing/invalid.

    Local-first knob: matches the env-override pattern used elsewhere (rerank/auth);
    non-numeric or non-positive values silently revert to the documented default.
    """

    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


# Follow-up context budget: keep injection small so it never crowds out retrieved
# evidence. Three continuity blocks (recent linear turns / semantic same-session
# recall / long-term memory) share one total char budget, filled by priority:
# recent -> semantic -> memory, each taking min(block cap, remaining total). All
# knobs are env-overridable; invalid values fall back to the default.
_CONTEXT_TURNS = _env_int("DEEPRESEARCH_CONTEXT_TURNS", 6)
_CONTEXT_MAX_CHARS = _env_int("DEEPRESEARCH_CONTEXT_MAX_CHARS", 6000)
_CONTEXT_RECENT_CHARS = _env_int("DEEPRESEARCH_CONTEXT_RECENT_CHARS", 3500)
_CONTEXT_SEMANTIC_CHARS = _env_int("DEEPRESEARCH_CONTEXT_SEMANTIC_CHARS", 1500)
_CONTEXT_MEMORY_CHARS = _env_int("DEEPRESEARCH_CONTEXT_MEMORY_CHARS", 1000)
_CONTEXT_MSG_CHARS = 1000


def _format_history(
    messages: list[dict],
    *,
    turns: int = _CONTEXT_TURNS,
    max_chars: int = _CONTEXT_MAX_CHARS,
    msg_chars: int = _CONTEXT_MSG_CHARS,
) -> str:
    """Render the most recent user/assistant turns as a compact transcript.

    Walks newest -> oldest so the latest turns survive the char budget; when the
    budget is exhausted only the most recent round(s) remain. Each message is
    whitespace-collapsed and truncated to ``msg_chars``.
    """

    recent = [
        m
        for m in messages
        if m.get("role") in ("user", "assistant") and (m.get("content") or "").strip()
    ][-turns:]
    lines: list[str] = []
    total = 0
    for message in reversed(recent):
        label = "User" if message["role"] == "user" else "Assistant"
        content = " ".join((message["content"] or "").split())
        if len(content) > msg_chars:
            content = content[:msg_chars].rstrip() + "…"
        line = f"{label}: {content}"
        if lines and total + len(line) > max_chars:
            break
        lines.append(line)
        total += len(line)
    lines.reverse()
    return "\n".join(lines)


def _evidence_from_checkpoint(value: dict) -> EvidenceItem:
    """Restore only the public EvidenceItem fields persisted by ``to_dict``."""

    return EvidenceItem(
        chunk_id=int(value.get("chunk_id") or 0),
        document_id=str(value.get("document_id") or ""),
        url=str(value.get("url") or ""),
        title=str(value.get("title") or ""),
        text=str(value.get("text") or ""),
        score=float(value.get("score") or 0.0),
        channels=[str(item) for item in (value.get("channels") or [])],
        origin=str(value.get("origin") or "external"),
        artifact_type=str(value.get("artifact_type") or "web_document"),
        artifact_id=str(value.get("artifact_id") or ""),
        grounding_eligible=bool(value.get("grounding_eligible", True)),
        provenance_refs=[
            str(item) for item in (value.get("provenance_refs") or [])
        ],
        source_status=str(value.get("source_status") or "active"),
        published_at=_optional_float(value.get("published_at")),
        updated_at=_optional_float(value.get("updated_at")),
        effective_at=_optional_float(value.get("effective_at")),
        retrieved_at=_optional_float(value.get("retrieved_at")),
        search_provider=str(value.get("search_provider") or ""),
        fetch_provider=str(value.get("fetch_provider") or ""),
    )


def _retrieval_outcome_from_checkpoint(value: dict) -> RetrievalOutcome:
    return RetrievalOutcome(
        lane_id=str(value.get("lane_id") or ""),
        lane=str(value.get("lane") or "external"),
        query=str(value.get("query") or ""),
        attempted=bool(value.get("attempted")),
        status=str(value.get("status") or "failed"),
        provider=(
            str(value["provider"]) if value.get("provider") is not None else None
        ),
        search_provider=(
            str(value["search_provider"])
            if value.get("search_provider") is not None
            else None
        ),
        fetch_providers=tuple(
            str(item) for item in (value.get("fetch_providers") or [])
        ),
        providers_attempted=tuple(
            str(item) for item in (value.get("providers_attempted") or [])
        ),
        providers_skipped=tuple(
            str(item) for item in (value.get("providers_skipped") or [])
        ),
        provider_failure_reasons={
            str(key): str(item)
            for key, item in (value.get("provider_failure_reasons") or {}).items()
        },
        tier=str(value["tier"]) if value.get("tier") is not None else None,
        hit_count=max(0, int(value.get("hit_count") or 0)),
        evidence_ids=tuple(
            str(item) for item in (value.get("evidence_ids") or [])
        ),
        error_code=(
            str(value["error_code"])
            if value.get("error_code") is not None
            else None
        ),
        reason=str(value["reason"]) if value.get("reason") is not None else None,
        locale=dict(value["locale"]) if isinstance(value.get("locale"), dict) else None,
        search_options=(
            dict(value["search_options"])
            if isinstance(value.get("search_options"), dict)
            else None
        ),
        started_at=float(value.get("started_at") or 0.0),
        finished_at=float(value.get("finished_at") or 0.0),
    )


def _remediation_from_checkpoint(value: dict) -> RemediationOutcome:
    return RemediationOutcome(
        attempted=bool(value.get("attempted")),
        attempt_index=max(0, int(value.get("attempt_index") or 0)),
        trigger=str(value["trigger"]) if value.get("trigger") is not None else None,
        action=str(value["action"]) if value.get("action") is not None else None,
        focused_query=str(value.get("focused_query") or ""),
        excluded_evidence_ids=tuple(
            str(item) for item in (value.get("excluded_evidence_ids") or [])
        ),
        excluded_provenance_roots=tuple(
            str(item)
            for item in (value.get("excluded_provenance_roots") or [])
        ),
        lane_ids=tuple(str(item) for item in (value.get("lane_ids") or [])),
        new_evidence_ids=tuple(
            str(item) for item in (value.get("new_evidence_ids") or [])
        ),
        status=str(value.get("status") or "not_attempted"),
        started_at=_optional_float(value.get("started_at")),
        finished_at=_optional_float(value.get("finished_at")),
        source_generation_id=(
            str(value["source_generation_id"])
            if value.get("source_generation_id") is not None
            else None
        ),
        replacement_generation_id=(
            str(value["replacement_generation_id"])
            if value.get("replacement_generation_id") is not None
            else None
        ),
    )


def _checkpoint_phase_metric(value: dict) -> dict:
    allowed = {
        "plan_id",
        "lane",
        "query",
        "stage",
        "provider",
        "attempt",
        "started_at",
        "finished_at",
        "duration_ms",
        "duration_s",
        "status",
        "deadline_remaining_before",
        "deadline_remaining_after",
        "candidate_evidence_ids",
        "evidence_ids",
        "reason_code",
    }
    return {
        key: _checkpoint_json_value(item)
        for key, item in value.items()
        if key in allowed
    }


def _checkpoint_conflict_decision(value: dict) -> dict:
    allowed = {
        "claim_id",
        "type",
        "claim_scope",
        "evidence_ids",
        "resolution",
        "reason",
        "basis",
        "winner_evidence_id",
    }
    return {
        key: _checkpoint_json_value(item)
        for key, item in value.items()
        if key in allowed
    }


def _checkpoint_json_value(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _checkpoint_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [_checkpoint_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return str(value)


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _sources(evidence: list[EvidenceItem]) -> list[dict]:
    return [e.to_dict() for e in evidence]


def _filter_lanes_to_final_sources(
    lanes: list[dict],
    final_evidence: list[EvidenceItem],
) -> list[dict]:
    final_keys = _final_source_keys(final_evidence)
    return [
        {
            **lane,
            "sources": [
                source
                for source in list(lane.get("sources") or [])
                if _source_dict_key(source) in final_keys
            ],
        }
        for lane in lanes
    ]


def _filter_sections_to_final_sources(
    sections: list[dict],
    final_evidence: list[EvidenceItem],
) -> list[dict]:
    final_keys = _final_source_keys(final_evidence)
    filtered: list[dict] = []
    for section in sections:
        answer = dict(section.get("answer") or {})
        citations = list(answer.get("citations") or [])
        if citations and any(_source_dict_key(citation) not in final_keys for citation in citations):
            continue
        filtered.append({key: value for key, value in section.items() if not key.startswith("_")})
    return filtered


def _filter_research_state_to_final_evidence(
    state_snapshot: dict,
    sections: list[dict],
    final_evidence: list[EvidenceItem],
) -> dict:
    """Remove notes derived from evidence excluded by the final snapshot."""

    safe_snapshot = dict(state_snapshot)
    safe_rows = [dict(row) for row in list(state_snapshot.get("evidence_matrix") or [])]
    final_keys = _final_source_keys(final_evidence)
    sections_by_row = {
        int(section["_matrix_index"]): section
        for section in sections
        if isinstance(section.get("_matrix_index"), int)
    }
    for index, row in enumerate(safe_rows):
        section = sections_by_row.get(index, {})
        evidence_keys = {
            str(value) for value in list(section.get("_evidence_keys") or []) if str(value)
        }
        if evidence_keys and not evidence_keys.issubset(final_keys):
            row["notes"] = ""
            row["source_count"] = 0
            row["status"] = "excluded_from_final"
    safe_snapshot["evidence_matrix"] = safe_rows
    return safe_snapshot


def _final_source_keys(evidence: list[EvidenceItem]) -> set[str]:
    return {
        key
        for item in evidence
        for key in (evidence_identity(item), provenance_family(item))
        if key
    }


def _source_dict_key(source: dict) -> str:
    refs = source.get("provenance_refs")
    if isinstance(refs, list) and refs:
        return str(refs[0])
    origin = str(source.get("origin") or "external")
    artifact_type = str(source.get("artifact_type") or "web_document")
    artifact_id = str(source.get("artifact_id") or "")
    if origin in {"local", "memory"} and artifact_id:
        return f"{origin}:{artifact_type}:{artifact_id}"
    document_id = str(source.get("document_id") or "")
    if document_id:
        return f"external:web_document:{document_id}"
    url = str(source.get("url") or "")
    return f"external:web_document:{url}" if url else ""


def _temporal_instruction(coverage: TemporalCoverage) -> str:
    if coverage.intent == "none":
        return ""
    parts = [
        f"Temporal evidence boundary: {coverage.statement}",
        "Treat effective_at as claim fact time, published_at/updated_at as source time only, and retrieved_at as retrieval time only; retrieved_at is not a fact-time signal.",
    ]
    if coverage.intent == "as_of":
        parts.append(
            "Historical as-of rule: state a claim as true at the requested date only when its cited evidence explicitly supports that date or an earlier state. Later-published material is eligible only when effective_at explicitly describes the requested historical state; otherwise present it only as a later boundary."
        )
    else:
        parts.append(
            "Latest/current rule: do not imply coverage beyond supported_through; disclose the nearest verifiable boundary without estimating missing current facts."
        )
    return " ".join(parts)


def _conflict_instruction(
    conflict_analysis: tuple[list, list[dict]],
) -> str:
    """Render bounded pre-composition conflict decisions for the answer writer."""

    _, decisions = conflict_analysis
    rules: list[str] = []
    for decision in decisions:
        evidence_ids = [str(value) for value in decision.get("evidence_ids", [])]
        citations = " and ".join(f"source {evidence_id}" for evidence_id in evidence_ids)
        if decision.get("resolution") == "resolved":
            winner = str(decision.get("winner_evidence_id") or "")
            basis = str(decision.get("basis") or "deterministic rules")
            if winner:
                rules.append(
                    f"Resolved evidence conflict {citations}: use [{winner}] as the controlling same-scope evidence based on {basis}."
                )
            else:
                rules.append(
                    f"Resolved evidence scope split {citations}: keep the statements in their distinct scopes and do not combine them."
                )
        else:
            rules.append(
                f"Unresolved evidence conflict {citations}: present both supported positions without choosing a winner."
            )
    if not rules:
        return ""
    return (
        "Evidence conflict rules:\n- "
        + "\n- ".join(rules)
        + "\nDo not add a conflict appendix inside the body; the product appends it after the body."
    )


def _grounding_repair_instruction(reason: str) -> str:
    """Return a deterministic, trusted instruction for the single T48 rewrite."""

    shape_rule = (
        "Satisfy the user's explicit sentence, length, citation-count, and output-shape constraints exactly. "
        if reason == "output_shape_invalid"
        else "Preserve all explicit user output-shape constraints. "
    )
    return (
        "Grounding repair: rewrite the answer from the supplied sources. "
        "Use one directly supported claim per sentence or bullet and place its [cN] marker on that same claim. "
        "Make every cited sentence independently understandable: repeat the named subject and any necessary "
        "scope instead of relying on pronouns or an antecedent from another sentence. "
        "For comparison answers, keep source-specific sides in separate sentences and use exactly one [cN] "
        "source per sentence; do not join complementary source claims with while, whereas, or compared directly. "
        "Cite multiple sources on one claim only when every cited source supports the complete claim. "
        "Paraphrase only predicates and attributes explicitly present in the cited source text; do not "
        "expand a source statement with related actions, benefits, or background knowledge. "
        "Remove unsupported synthesis, repeated filler, and claims that rely on source titles alone. "
        "Finish every sentence and every required section. "
        + shape_rule
    )


def _grounding_repair_action(
    *,
    reason: str,
    unknown_citations: int,
    citation_binding_valid: bool,
    mode: Mode,
    evidence_count: int,
) -> str:
    """Choose the one T48 repair branch without weakening grounding thresholds."""

    if reason in {"invalid_citation_binding", "output_shape_invalid"} or unknown_citations > 0:
        return "rewrite_only"
    if reason == "coverage_below_threshold":
        minimum_for_rewrite = 2 if mode in {Mode.MULTIPLE, Mode.DEEP} else 1
        return "rewrite_only" if evidence_count >= minimum_for_rewrite else "retrieve_and_rewrite"
    if not citation_binding_valid:
        return "rewrite_only"
    return "retrieve_and_rewrite"


def _combine_instructions(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if part and part.strip())


def _dedup_evidence(items: list[EvidenceItem]) -> list[EvidenceItem]:
    best: dict[str, EvidenceItem] = {}
    for item in items:
        key = provenance_family(item)
        if key not in best or item.score > best[key].score:
            best[key] = item
    return sorted(best.values(), key=lambda i: i.score, reverse=True)


def _exclude_evidence(
    items: list[EvidenceItem],
    *,
    evidence_ids: tuple[str, ...],
    provenance_roots: tuple[str, ...],
) -> list[EvidenceItem]:
    """Exclude only run-scoped rejected evidence during T48 recovery."""

    excluded_ids = set(evidence_ids)
    excluded_roots = set(provenance_roots)
    return [
        item
        for item in items
        if evidence_identity(item) not in excluded_ids and provenance_family(item) not in excluded_roots
    ]


def _fuse_evidence_rankings(
    rankings: list[list[EvidenceItem]],
    top_k: int,
    *,
    preserve_ranking_diversity: bool = False,
    merge_complementary_text: bool = False,
) -> list[EvidenceItem]:
    """Fuse incomparable lane rankings by rank while preserving lane coverage."""

    limit = max(0, int(top_k))
    if limit == 0:
        return []
    scores: dict[str, float] = {}
    occurrences: dict[str, int] = {}
    representatives: dict[str, EvidenceItem] = {}
    ranking_keys: list[list[str]] = []
    for ranking in rankings:
        keys: list[str] = []
        seen: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            key = provenance_family(item)
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
            rank_score = 1.0 / (60 + rank)
            if preserve_ranking_diversity:
                # Repeated retrieval of the same provenance family across
                # related subqueries is not independent corroboration.
                scores[key] = max(scores.get(key, 0.0), rank_score)
            else:
                scores[key] = scores.get(key, 0.0) + rank_score
            occurrences[key] = occurrences.get(key, 0) + 1
            if key in representatives:
                merge = (
                    _merge_complementary_evidence_representatives
                    if merge_complementary_text
                    else _merge_evidence_representatives
                )
                representatives[key] = merge(representatives[key], item)
            else:
                representatives[key] = item
        if keys:
            ranking_keys.append(keys)
    if not representatives:
        return []

    selected: list[str] = []
    selected_set: set[str] = set()
    if limit >= len(ranking_keys):
        for keys in ranking_keys:
            available = [candidate for candidate in keys if candidate not in selected_set]
            if preserve_ranking_diversity and available:
                rarest = min(occurrences[candidate] for candidate in available)
                key = next(
                    candidate
                    for candidate in available
                    if occurrences[candidate] == rarest
                )
            else:
                key = available[0] if available else None
            if key is not None:
                selected.append(key)
                selected_set.add(key)

    ordered = sorted(
        representatives,
        key=lambda key: (
            -scores[key],
            representatives[key].title,
            representatives[key].text,
            key,
        ),
    )
    for key in ordered:
        if len(selected) >= limit:
            break
        if key in selected_set:
            continue
        selected.append(key)
        selected_set.add(key)

    selected.sort(
        key=lambda key: (
            -scores[key],
            representatives[key].title,
            representatives[key].text,
            key,
        )
    )
    return [replace(representatives[key], score=scores[key]) for key in selected]


def _merge_evidence_representatives(
    current: EvidenceItem,
    candidate: EvidenceItem,
) -> EvidenceItem:
    """Merge duplicate-family evidence without preferring any provenance origin."""

    def quality(item: EvidenceItem) -> tuple[int, int, int, float, int, int]:
        source_date_fields = sum(
            value is not None for value in (item.published_at, item.updated_at)
        )
        return (
            int(item.effective_at is not None),
            source_date_fields,
            int(item.retrieved_at is not None),
            float(item.retrieved_at or 0.0),
            len(item.provenance_refs),
            len(item.text),
        )

    preferred, fallback = (
        (candidate, current)
        if quality(candidate) > quality(current)
        else (current, candidate)
    )
    return replace(
        preferred,
        document_id=preferred.document_id or fallback.document_id,
        url=preferred.url or fallback.url,
        channels=list(dict.fromkeys([*preferred.channels, *fallback.channels])),
        artifact_id=preferred.artifact_id or fallback.artifact_id,
        provenance_refs=list(
            dict.fromkeys([*preferred.provenance_refs, *fallback.provenance_refs])
        ),
    )


def _merge_complementary_evidence_representatives(
    current: EvidenceItem,
    candidate: EvidenceItem,
) -> EvidenceItem:
    """Join same-document excerpts without combining distinct temporal snapshots."""

    merged = _merge_evidence_representatives(current, candidate)
    same_document_id = bool(
        current.document_id
        and candidate.document_id
        and current.document_id == candidate.document_id
    )
    same_canonical_source = bool(
        current.url
        and candidate.url
        and current.url == candidate.url
        and provenance_family(current) == provenance_family(candidate)
    )
    same_document = same_document_id or same_canonical_source
    temporal_fields = ("effective_at", "published_at", "updated_at")
    same_temporal_snapshot = all(
        getattr(current, field_name) == getattr(candidate, field_name)
        for field_name in temporal_fields
    )
    if not same_document or not same_temporal_snapshot:
        return merged

    paragraphs: list[str] = []
    seen: set[str] = set()
    for text in (current.text, candidate.text):
        for paragraph in re.split(r"\n\s*\n", str(text or "")):
            cleaned = paragraph.strip()[:_COMPLEMENTARY_EVIDENCE_PIECE_CHARS].strip()
            if cleaned and cleaned not in seen:
                paragraphs.append(cleaned)
                seen.add(cleaned)

    if len(paragraphs) < 2:
        return merged

    combined = ""
    for paragraph in paragraphs:
        separator = "\n\n" if combined else ""
        remaining = _COMPLEMENTARY_EVIDENCE_MAX_CHARS - len(combined) - len(separator)
        if remaining <= 0:
            break
        combined += separator + paragraph[:remaining]
    return replace(
        merged,
        text=combined or merged.text,
        channels=list(
            dict.fromkeys(
                [*merged.channels, COMPLEMENTARY_EVIDENCE_CHANNEL]
            )
        ),
    )


def _merge_extra_subqueries(subqueries: list[str], *, count: int, options: SearchOptions | None) -> list[str]:
    """Append user subqueries after LLM ones, preserving order and budget."""

    merged: list[str] = []
    seen: set[str] = set()
    for query in [*subqueries, *((options.extra_subqueries if options else ()))]:
        cleaned = " ".join(str(query or "").split())
        key = cleaned.lower()
        if len(cleaned) > 3 and key not in seen:
            merged.append(cleaned)
            seen.add(key)
        if len(merged) >= max(1, count):
            break
    return merged or [subqueries[0] if subqueries else ""]


_EN_COMPARISON_FOCUS = re.compile(
    r"^\s*(?:compare|contrast)(?:\s+between)?\s+(.+?)\s+"
    r"(?:and|with|versus|vs\.?)\s+(.+?)\s*[.?!]*$",
    re.IGNORECASE,
)
_ZH_COMPARISON_FOCUS = re.compile(
    r"^\s*(?:比较|对比)\s*(.+?)\s*(?:和|与|及|vs\.?|versus)\s*(.+?)\s*[。？！.?!]*$",
    re.IGNORECASE,
)


def _source_constrained_subqueries(
    intent: ResearchIntent,
    subqueries: list[str],
    *,
    count: int,
) -> list[str]:
    """Prioritize one source-specific query per side of an explicit comparison.

    A broad comparison query can rank documents for only one entity even when the
    user explicitly requires authoritative or primary sources.  This deterministic
    projection is derived only from the typed research focus; it does not impose a
    provider/domain priority and leaves unconstrained/non-comparison queries intact.
    """

    maximum = max(1, int(count))
    if intent.source_requirement == "none" or maximum < 2:
        return list(subqueries[:maximum])
    focus = " ".join(intent.research_focus.split()).strip()
    match = _EN_COMPARISON_FOCUS.match(focus) or _ZH_COMPARISON_FOCUS.match(focus)
    if match is None:
        return list(subqueries[:maximum])
    subjects = [
        " ".join(value.split()).strip(" .?!。？！")
        for value in match.groups()
    ]
    if any(len(subject) < 2 or len(subject) > 160 for subject in subjects):
        return list(subqueries[:maximum])
    if subjects[0].casefold() == subjects[1].casefold():
        return list(subqueries[:maximum])

    cjk_focus = bool(re.search(r"[\u3400-\u9fff]", focus))
    if cjk_focus:
        suffix = "第一手资料" if intent.source_requirement == "primary" else "官方文档"
    else:
        suffix = "primary source" if intent.source_requirement == "primary" else "official documentation"
    prioritized = [f"{subject} {suffix}" for subject in subjects]

    merged: list[str] = []
    seen: set[str] = set()
    for query in [*prioritized, *subqueries]:
        cleaned = " ".join(str(query or "").split())
        key = cleaned.casefold()
        if len(cleaned) > 3 and key not in seen:
            merged.append(cleaned)
            seen.add(key)
        if len(merged) >= maximum:
            break
    return merged or list(subqueries[:maximum])


def _stats_search_tier(stats: list[dict]) -> str | None:
    for item in stats:
        tier = item.get("fallback_tier")
        if isinstance(tier, str) and tier:
            return tier
    return None


def _stats_search_providers(stats: list[dict]) -> set[str]:
    providers: set[str] = set()
    for item in stats:
        for key in ("search_provider", "fetch_provider"):
            value = item.get(key)
            if value in {"brave", "tavily", "exa", "firecrawl"}:
                providers.add(value)
    return providers


def _external_provider(stats: list[dict]) -> str | None:
    providers = sorted(_stats_search_providers(stats))
    return "+".join(providers) if providers else None


def _latest_external_tier(outcomes: list[RetrievalOutcome]) -> str | None:
    for outcome in reversed(outcomes):
        if outcome.lane == "external" and outcome.tier:
            return outcome.tier
    return None


def _adapter_model_ref(adapter: object) -> str:
    value = str(getattr(adapter, "_deepresearch_model_ref", "") or "")
    return value if re.fullmatch(r"[A-Za-z0-9._:/-]{1,160}", value) else ""


def _stats_degraded_fetch(stats: list[dict]) -> bool:
    return any(bool(item.get("degraded_fetch")) for item in stats)


def _search_usage_kwargs(search: object | None, request: RunRequest) -> dict[str, object]:
    """Pass the frozen policy only to T50-aware adapters, preserving test adapters."""

    if search is None or not getattr(search, "supports_search_usage_policy", False):
        return {}
    return {"resolved_usage": dict(request.resolved_search_usage)}


def _provider_circuit_kwargs(search: object | None, ctx: _RunContext) -> dict[str, object]:
    """Pass a run-local auth/payment circuit only to T50-aware adapters."""

    if search is None or not getattr(search, "supports_search_usage_policy", False):
        return {}
    return {"provider_circuit": ctx.provider_circuit_open}


def _search_relevance_kwargs(
    search: object | None,
    relevance_query: str,
) -> dict[str, object]:
    """Filter legacy cache hits only for adapters that implement the T51 contract."""

    if search is None or not getattr(search, "supports_relevance_aware_fallback", False):
        return {}
    return {"relevance_query": relevance_query}


def _focused_fetch_kwargs(
    search: object | None,
    relevance_query: str,
) -> dict[str, object]:
    """Pass a research focus only to adapters with query-isolated fetch support."""

    if search is None or not getattr(search, "supports_focused_fetch", False):
        return {}
    return {"relevance_query": relevance_query}


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def _writer_meta_payload(
    metadata: StreamComposeMeta,
    *,
    generation_id: str | None = None,
) -> dict[str, object]:
    attempts = [dict(item) for item in metadata.attempts]
    if generation_id:
        for attempt in attempts:
            attempt["generation_id"] = generation_id
    payload: dict[str, object] = {
        "classification": metadata.classification or (
            "completed" if metadata.visible_chars > 0 else "empty_stream"
        ),
        "provider": metadata.provider,
        "model": metadata.model,
        "finish_reason": metadata.finish_reason,
        "visible_chars": max(0, int(metadata.visible_chars)),
        "usage": dict(metadata.usage),
        "fallback_scope": metadata.fallback_scope,
        "sync_fallback_used": bool(metadata.sync_fallback_used),
        "attempts": attempts,
        "call_options": dict(metadata.call_options),
        "generation_id": generation_id,
    }
    return _safe_writer_diagnostics(payload)


def _merge_writer_diagnostics(
    previous: dict[str, object],
    current: dict[str, object],
) -> dict[str, object]:
    if not previous:
        return dict(current)
    merged = dict(current)
    prior_attempts = previous.get("attempts")
    current_attempts = current.get("attempts")
    merged["attempts"] = [
        *(
            [dict(item) for item in prior_attempts if isinstance(item, dict)]
            if isinstance(prior_attempts, list)
            else []
        ),
        *(
            [dict(item) for item in current_attempts if isinstance(item, dict)]
            if isinstance(current_attempts, list)
            else []
        ),
    ]
    return merged


def _safe_writer_diagnostics(raw: dict) -> dict[str, object]:
    """Keep the public writer audit free of prompts, keys and reasoning content."""

    safe: dict[str, object] = {}
    for key in (
        "classification",
        "provider",
        "model",
        "finish_reason",
        "fallback_scope",
        "generation_id",
    ):
        value = raw.get(key)
        if value is None:
            safe[key] = None
        elif isinstance(value, (str, int, float, bool)):
            safe[key] = _safe_diagnostic_scalar(value)
    for key in ("visible_chars",):
        try:
            safe[key] = max(0, int(raw.get(key) or 0))
        except (TypeError, ValueError):
            safe[key] = 0
    if isinstance(raw.get("sync_fallback_used"), bool):
        safe["sync_fallback_used"] = raw["sync_fallback_used"]
    usage = raw.get("usage")
    if isinstance(usage, dict):
        safe["usage"] = {
            str(key)[:80]: value
            for key, value in usage.items()
            if isinstance(value, (int, float, bool)) or value is None
        }
    attempts = raw.get("attempts")
    if isinstance(attempts, (list, tuple)):
        safe_attempts: list[dict[str, object]] = []
        for item in attempts:
            if not isinstance(item, dict):
                continue
            safe_attempt: dict[str, object] = {}
            for key in (
                "attempt",
                "provider",
                "model",
                "call_mode",
                "classification",
                "status",
                "fallback_scope",
                "finish_reason",
                "visible_chars",
                "generation_id",
            ):
                if key not in item:
                    continue
                value = item.get(key)
                if value is None or isinstance(value, (str, int, float, bool)):
                    safe_attempt[key] = _safe_diagnostic_scalar(value)
            item_usage = item.get("usage")
            if isinstance(item_usage, dict):
                safe_attempt["usage"] = {
                    str(key)[:80]: value
                    for key, value in item_usage.items()
                    if isinstance(value, (int, float, bool)) or value is None
                }
            item_options = item.get("call_options")
            if isinstance(item_options, dict):
                safe_attempt["call_options"] = _safe_writer_diagnostics(
                    {"call_options": item_options}
                ).get("call_options", {})
            safe_attempts.append(safe_attempt)
        safe["attempts"] = safe_attempts
    call_options = raw.get("call_options")
    if isinstance(call_options, dict):
        safe_options: dict[str, object] = {}
        for key in (
            "provider_identity",
            "provider_family",
            "endpoint_type",
            "reasoning_transport",
            "structured_output_strategy",
            "output_token_parameter",
            "shared_output_cap",
            "thinking_target",
            "visible_output_target",
            "reasoning_reserve_tokens",
            "provider_output_cap",
        ):
            value = call_options.get(key)
            if isinstance(value, (str, int, float, bool)) or value is None:
                safe_options[key] = _safe_diagnostic_scalar(value)
        for key in ("requested", "applied", "ignored"):
            value = call_options.get(key)
            if isinstance(value, dict):
                safe_options[key] = {
                    str(item_key)[:80]: _safe_diagnostic_scalar(item_value)
                    for item_key, item_value in value.items()
                    if isinstance(item_value, (str, int, float, bool)) or item_value is None
                }
        reason_codes = call_options.get("reason_codes")
        if isinstance(reason_codes, (list, tuple)):
            safe_options["reason_codes"] = [
                _safe_diagnostic_scalar(item, limit=120)
                for item in reason_codes
                if isinstance(item, str)
            ]
        safe["call_options"] = safe_options
    return safe


def _safe_diagnostic_scalar(value: object, *, limit: int = 200) -> object:
    if not isinstance(value, str):
        return value
    redacted = redact_text(value)
    return re.sub(r"[\r\n\t]+", " ", redacted)[: max(1, int(limit))]


_SUBQUERY_PROMPT = "Break the research question into {n} distinct search sub-queries, one per line:\n\n{query}"
_PLAN_PROMPT = "Create a concise research outline (3-5 section titles, one per line) for:\n\n{query}"
_PLAN_COUNT_PROMPT = (
    "Create a concise research outline with exactly {n} non-redundant section titles, one per line. "
    "Together they must cover every topic explicitly requested in:\n\n{query}"
)
_GAP_PROMPT = (
    "GAP-ANALYSIS. Research question:\n{query}\n\nSections drafted so far:\n{draft}\n\n"
    "List up to 3 important sub-questions still unanswered, one per line. "
    "If the draft is already comprehensive, reply with exactly NONE."
)
# Lines that mean "no further gaps" (includes the FakeLLM/no-grounding sentinel).
# Compared after stripping list markers and trailing punctuation, so no trailing period here.
_GAP_SENTINELS = {"none", "no gaps", "no grounded answer available"}
