"""Deadline-independent semantic enrichment for persisted continuity rows.

This module intentionally knows nothing about research run state or SSE.  It
operates only on already-committed message/session-memory rows through a small
store protocol, which keeps post-terminal enrichment idempotent and prevents a
maintenance failure from mutating a completed run.
"""

from __future__ import annotations

import hashlib
import math
import queue
import threading
import time
import uuid
import urllib.error
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import Any, Callable, ContextManager, Iterable, Mapping, Protocol, Sequence

from .deadline import ProviderCallBudget, use_provider_call_budget
from .errors import BaselineError, ErrorCode
from ..observability import get_logger


_LOG = get_logger("deepresearch.semantic_enrichment")


SEMANTIC_QUEUE_NAME = "deepresearch-semantic"
RESEARCH_QUEUE_NAME = "deepresearch"
MAINTENANCE_BUDGET_S = 30.0
TRANSPORT_TIMEOUT_S = 10.0
MAINTENANCE_CLEANUP_GRACE_S = 5.0
SEMANTIC_JOB_TIMEOUT_S = int(MAINTENANCE_BUDGET_S + MAINTENANCE_CLEANUP_GRACE_S)
MAX_PROVIDER_ATTEMPTS = 2
MAX_BATCH_ROWS = 32
MAX_BATCH_BYTES = 64 * 1024
MAX_ITEM_BYTES = 16 * 1024
THREAD_PENDING_MAX = 128
SEMANTIC_OUTCOME_STATUSES = frozenset(
    {"completed", "empty", "deferred", "skipped", "timeout", "failed"}
)


@dataclass(frozen=True)
class MissingEmbeddingRow:
    """One persisted continuity row whose embedding is still missing."""

    kind: str
    id: str
    session_id: str
    content: str
    content_sha256: str
    created_at: float

    @classmethod
    def from_value(cls, value: "MissingEmbeddingRow | Mapping[str, Any]") -> "MissingEmbeddingRow":
        if isinstance(value, cls):
            return value
        content = str(value.get("content") or "")
        return cls(
            kind=str(value.get("kind") or ""),
            id=str(value.get("id") or ""),
            session_id=str(value.get("session_id") or ""),
            content=content,
            content_sha256=str(value.get("content_sha256") or _content_sha256(content)),
            created_at=float(value.get("created_at") or 0.0),
        )


@dataclass(frozen=True)
class SemanticEnrichmentRequest:
    """Secret-free maintenance request suitable for an in-process or RQ queue."""

    session_id: str
    run_id: str = ""
    trigger: str = "post_run"
    attempt: int = 0

    def __post_init__(self) -> None:
        if self.trigger not in {"post_run", "backlog"}:
            raise ValueError("trigger must be 'post_run' or 'backlog'")
        if not self.session_id:
            raise ValueError("session_id is required")
        if self.attempt < 0:
            raise ValueError("attempt must be non-negative")

    def to_payload(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "run_id": self.run_id,
            "trigger": self.trigger,
            "attempt": self.attempt,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "SemanticEnrichmentRequest":
        return cls(
            session_id=str(payload.get("session_id") or ""),
            run_id=str(payload.get("run_id") or ""),
            trigger=str(payload.get("trigger") or "post_run"),
            attempt=max(0, int(payload.get("attempt") or 0)),
        )


@dataclass(frozen=True)
class SemanticEmbeddingWrite:
    """Content-aware compare-and-set request for one embedding."""

    kind: str
    id: str
    content: str
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class SemanticEnrichmentOutcome:
    """Secret-free result of one semantic maintenance job."""

    status: str
    session_id: str
    run_id: str = ""
    trigger: str = "post_run"
    scanned_count: int = 0
    written_count: int = 0
    skipped_count: int = 0
    batch_count: int = 0
    provider_attempts: int = 0
    duration_ms: float = 0.0
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in SEMANTIC_OUTCOME_STATUSES:
            raise ValueError("invalid semantic enrichment status")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "session_id": self.session_id,
            "run_id": self.run_id or None,
            "trigger": self.trigger,
            "scanned_count": self.scanned_count,
            "written_count": self.written_count,
            "skipped_count": self.skipped_count,
            "batch_count": self.batch_count,
            "provider_attempts": self.provider_attempts,
            "duration_ms": round(max(0.0, self.duration_ms), 3),
            "reason_codes": list(self.reason_codes),
        }


class SemanticEnrichmentStore(Protocol):
    """Store surface required by :func:`enrich_semantic_continuity`."""

    def is_session_active(self, session_id: str) -> bool: ...

    def list_missing_embedding_rows(
        self,
        session_id: str,
        *,
        after_created_at: float | None = None,
        after_id: str | None = None,
        limit: int = MAX_BATCH_ROWS,
    ) -> Sequence[MissingEmbeddingRow | Mapping[str, Any]]: ...

    def compare_and_set_embedding(
        self,
        kind: str,
        row_id: str,
        expected_content: str,
        embedding: Sequence[float],
    ) -> bool: ...

    def compare_and_set_embeddings(
        self,
        updates: Sequence[tuple[str, str, str, Sequence[float]]],
    ) -> int | Sequence[str] | set[str]: ...


class EmbeddingAdapter(Protocol):
    def embed(self, texts: Sequence[str]) -> Any: ...


ResearchBusyCheck = Callable[[], bool]
LeaseFactory = Callable[[str], ContextManager[bool]]
Clock = Callable[[], float]
Sleeper = Callable[[float], None]
OutcomeSink = Callable[[SemanticEnrichmentOutcome], None]
EmbedderResolver = Callable[[], EmbeddingAdapter | None]


class SemanticEnricher:
    """Fill missing continuity embeddings under an independent maintenance budget."""

    def __init__(
        self,
        store: SemanticEnrichmentStore,
        embedder: EmbeddingAdapter | None,
        research_busy: ResearchBusyCheck | None = None,
        lease: LeaseFactory | None = None,
        *,
        embedder_resolver: EmbedderResolver | None = None,
        outcome_sink: OutcomeSink | None = None,
        clock: Clock = time.monotonic,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._research_busy = research_busy
        self._lease = lease
        self._embedder_resolver = embedder_resolver
        self._outcome_sink = outcome_sink
        self._clock = clock
        self._sleeper = sleeper

    def run(self, request: SemanticEnrichmentRequest) -> SemanticEnrichmentOutcome:
        """Run one synchronous enrichment job without touching run state or SSE."""

        return _run_enrichment(
            store=self._store,
            embedder=self._embedder,
            request=request,
            research_busy=self._research_busy,
            lease_factory=self._lease,
            embedder_resolver=self._embedder_resolver,
            outcome_sink=self._outcome_sink,
            clock=self._clock,
            sleeper=self._sleeper,
        )


def _run_enrichment(
    *,
    store: SemanticEnrichmentStore,
    embedder: EmbeddingAdapter | None,
    request: SemanticEnrichmentRequest,
    research_busy: ResearchBusyCheck | None,
    lease_factory: LeaseFactory | None,
    embedder_resolver: EmbedderResolver | None,
    outcome_sink: OutcomeSink | None,
    clock: Clock,
    sleeper: Sleeper,
) -> SemanticEnrichmentOutcome:

    started_at = clock()
    deadline = started_at + MAINTENANCE_BUDGET_S
    reasons: list[str] = []
    scanned = written = skipped = batches = provider_attempts = 0

    def finish(status: str, *extra_reasons: str) -> SemanticEnrichmentOutcome:
        outcome = SemanticEnrichmentOutcome(
            status=status,
            session_id=request.session_id,
            run_id=request.run_id,
            trigger=request.trigger,
            scanned_count=scanned,
            written_count=written,
            skipped_count=skipped,
            batch_count=batches,
            provider_attempts=provider_attempts,
            duration_ms=max(0.0, (clock() - started_at) * 1000.0),
            reason_codes=_unique((*reasons, *extra_reasons)),
        )
        if outcome_sink is not None:
            outcome_sink(outcome)
        return outcome

    if embedder is None and embedder_resolver is None:
        return finish("skipped", "skipped_no_embedder")
    if research_busy is not None:
        try:
            if research_busy():
                return finish("deferred", "deferred_research_busy")
        except Exception:  # noqa: BLE001 - queue observability is maintenance-only
            return finish("failed", "research_queue_check_failed")
    if not _session_active(store, request.session_id):
        return finish("skipped", "skipped_session_inactive")

    lease_context = (
        lease_factory(request.session_id)
        if lease_factory is not None
        else nullcontext(True)
    )
    with lease_context as acquired:
        if not acquired:
            return finish("deferred", "deferred_session_lease_busy")

        cursor_created_at: float | None = None
        cursor_id: str | None = None
        while True:
            if clock() >= deadline:
                return finish("timeout", "maintenance_deadline_exceeded")
            if research_busy is not None:
                try:
                    if research_busy():
                        return finish("deferred", "deferred_research_busy")
                except Exception:  # noqa: BLE001 - queue observability is maintenance-only
                    return finish("failed", "research_queue_check_failed")
            if not _session_active(store, request.session_id):
                return finish("skipped", "skipped_session_inactive")

            rows = _list_missing_rows(
                store,
                request.session_id,
                after_created_at=cursor_created_at,
                after_id=cursor_id,
                limit=MAX_BATCH_ROWS,
            )
            if not rows:
                return finish("completed" if scanned else "empty")

            try:
                batch_embedder = (
                    embedder_resolver()
                    if embedder_resolver is not None
                    else embedder
                )
            except Exception:  # noqa: BLE001 - maintenance resolution is isolated
                return finish("failed", "embedder_resolution_failed")
            if batch_embedder is None:
                return finish("skipped", "skipped_no_embedder")

            batch_rows, texts, truncated = build_embedding_batch(rows)
            if not batch_rows:
                return finish("failed", "invalid_missing_embedding_row")
            scanned += len(batch_rows)
            if truncated:
                reasons.append("content_truncated_for_embedding")
            cursor_created_at = batch_rows[-1].created_at
            cursor_id = batch_rows[-1].id
            # The row scan and embedder resolution may yield to an archive/delete
            # request. Re-check at the final boundary before any content leaves the
            # process; the post-call check below remains the no-resurrection fence.
            if not _session_active(store, request.session_id):
                return finish("skipped", "skipped_session_inactive")

            try:
                vectors, attempts = _embed_with_budget(
                    batch_embedder,
                    texts,
                    deadline=deadline,
                    clock=clock,
                    sleeper=sleeper,
                )
                provider_attempts += attempts
                if attempts > 1:
                    reasons.append("provider_transient_retry")
            except BaselineError as error:
                provider_attempts += int(getattr(error, "_semantic_attempts", 1))
                if error.code == ErrorCode.RUN_TIMEOUT or clock() >= deadline:
                    return finish("timeout", "maintenance_deadline_exceeded")
                return finish("failed", _provider_failure_reason(error))
            except Exception as error:  # noqa: BLE001 - normalize provider-specific failures
                provider_attempts += int(getattr(error, "_semantic_attempts", 1))
                if clock() >= deadline:
                    return finish("timeout", "maintenance_deadline_exceeded")
                return finish("failed", _provider_failure_reason(error))

            try:
                normalized = validate_embedding_vectors(vectors, expected=len(batch_rows))
            except ValueError:
                return finish("failed", "invalid_embedding_response")

            if not _session_active(store, request.session_id):
                return finish("skipped", "skipped_session_inactive")

            writes = [
                SemanticEmbeddingWrite(
                    kind=row.kind,
                    id=row.id,
                    content=row.content,
                    embedding=tuple(vector),
                )
                for row, vector in zip(batch_rows, normalized, strict=True)
            ]
            try:
                batch_written = _compare_and_set_batch(store, writes)
            except Exception:  # noqa: BLE001 - maintenance must not affect terminal runs
                return finish("failed", "failed_internal")
            batches += 1
            batch_written = max(0, min(len(writes), int(batch_written)))
            written += batch_written
            stale = len(writes) - batch_written
            skipped += stale
            if stale:
                reasons.append("stale_content")


def validate_embedding_vectors(response: Any, expected: int) -> list[list[float]]:
    """Validate and order an embedding response without permitting partial binds."""

    if expected <= 0:
        raise ValueError("expected must be positive")
    payload = response
    if isinstance(response, Mapping) and "data" in response:
        payload = response.get("data")
    if not isinstance(payload, Sequence) or isinstance(payload, (str, bytes, bytearray)):
        raise ValueError("embedding response must be a sequence")
    if len(payload) != expected:
        raise ValueError("embedding response count mismatch")

    indexed = any(isinstance(item, Mapping) and "index" in item for item in payload)
    vectors: list[Any]
    if indexed:
        by_index: dict[int, Any] = {}
        for item in payload:
            if not isinstance(item, Mapping) or "index" not in item:
                raise ValueError("mixed indexed and unindexed embeddings")
            index = item.get("index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise ValueError("embedding index must be an integer")
            if index < 0 or index >= expected or index in by_index:
                raise ValueError("embedding index is duplicate or out of range")
            by_index[index] = item.get("embedding")
        if set(by_index) != set(range(expected)):
            raise ValueError("embedding indices are incomplete")
        vectors = [by_index[index] for index in range(expected)]
    else:
        vectors = [
            item.get("embedding") if isinstance(item, Mapping) else item
            for item in payload
        ]

    normalized: list[list[float]] = []
    dimension: int | None = None
    for vector in vectors:
        if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes, bytearray)):
            raise ValueError("embedding vector must be a sequence")
        parsed: list[float] = []
        for value in vector:
            if isinstance(value, bool):
                raise ValueError("embedding components must be numeric")
            try:
                number = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError("embedding components must be numeric") from error
            if not math.isfinite(number):
                raise ValueError("embedding components must be finite")
            parsed.append(number)
        if not parsed or not any(number != 0.0 for number in parsed):
            raise ValueError("embedding vector must be non-zero")
        if dimension is None:
            dimension = len(parsed)
        elif len(parsed) != dimension:
            raise ValueError("embedding dimensions must match")
        normalized.append(parsed)
    return normalized


def _embed_with_budget(
    embedder: EmbeddingAdapter,
    texts: Sequence[str],
    *,
    deadline: float,
    clock: Clock,
    sleeper: Sleeper,
) -> tuple[Any, int]:
    base_budget = ProviderCallBudget(
        request_deadline=deadline,
        phase_deadline=deadline,
        max_attempts=MAX_PROVIDER_ATTEMPTS,
        clock=clock,
    )
    last_error: Exception | None = None
    attempts = 0
    for attempt in range(MAX_PROVIDER_ATTEMPTS):
        attempt_budget = replace(
            base_budget,
            phase_deadline=min(deadline, clock() + TRANSPORT_TIMEOUT_S),
            attempt=attempt,
            max_attempts=attempt + 1,
        )
        attempt_budget.check("semantic_embedding_start")
        attempts += 1
        try:
            with use_provider_call_budget(attempt_budget):
                response = embedder.embed(texts)
            attempt_budget.check("semantic_embedding_finish")
            return response, attempts
        except Exception as error:  # noqa: BLE001 - providers expose several error types
            last_error = error
            retry_budget = base_budget.for_attempt(attempt)
            if not _is_transient(error) or not retry_budget.retry_allowed(0.1):
                _attach_attempt_count(error, attempts)
                raise
            retry_budget.sleep(0.1, sleeper=sleeper, stage="semantic_embedding_retry")
    assert last_error is not None
    _attach_attempt_count(last_error, attempts)
    raise last_error


def _is_transient(error: Exception) -> bool:
    if isinstance(error, BaselineError):
        return error.code in {ErrorCode.UPSTREAM_ERROR, ErrorCode.RUN_TIMEOUT}
    retryable = getattr(error, "retryable", None)
    if retryable is not None:
        return bool(retryable)
    if isinstance(error, urllib.error.HTTPError):
        return error.code == 429 or 500 <= error.code <= 599
    return isinstance(error, (urllib.error.URLError, TimeoutError, ConnectionError, OSError))


def _provider_failure_reason(error: Exception) -> str:
    status_code = getattr(error, "status_code", None)
    if not isinstance(status_code, int) and isinstance(error, urllib.error.HTTPError):
        status_code = int(error.code)
    if status_code in {401, 403}:
        return "credential_rejected"
    if status_code in {402, 432, 433}:
        return "provider_action_required"
    if status_code == 429:
        return "rate_limited"
    if isinstance(status_code, int) and status_code >= 500:
        return "service_unavailable"
    if isinstance(error, TimeoutError):
        return "provider_timeout"
    if isinstance(error, urllib.error.URLError):
        return "network_unreachable"
    return "failed_provider"


def _attach_attempt_count(error: Exception, attempts: int) -> None:
    try:
        setattr(error, "_semantic_attempts", attempts)
    except Exception:  # noqa: BLE001 - some exception types disallow attributes
        pass


def build_embedding_batch(
    rows: Sequence[MissingEmbeddingRow],
) -> tuple[list[MissingEmbeddingRow], list[str], bool]:
    """Select the next deterministic row/byte-bounded embedding batch."""

    selected: list[MissingEmbeddingRow] = []
    texts: list[str] = []
    total = 0
    truncated = False
    for row in rows[:MAX_BATCH_ROWS]:
        if not row.id or not row.kind or not row.content:
            continue
        text, item_truncated = _embedding_text(row.content)
        size = len(text.encode("utf-8"))
        if selected and total + size > MAX_BATCH_BYTES:
            break
        selected.append(row)
        texts.append(text)
        total += size
        truncated = truncated or item_truncated
    return selected, texts, truncated


def _embedding_text(content: str) -> tuple[str, bool]:
    encoded = content.encode("utf-8")
    if len(encoded) <= MAX_ITEM_BYTES:
        return content, False
    head_size = MAX_ITEM_BYTES // 2
    tail_size = MAX_ITEM_BYTES - head_size
    head = encoded[:head_size].decode("utf-8", errors="ignore")
    tail = encoded[-tail_size:].decode("utf-8", errors="ignore")
    value = f"{head}\n[…]\n{tail}"
    while len(value.encode("utf-8")) > MAX_ITEM_BYTES:
        tail = tail[1:]
        value = f"{head}\n[…]\n{tail}"
    return value, True


def _list_missing_rows(
    store: SemanticEnrichmentStore,
    session_id: str,
    *,
    after_created_at: float | None,
    after_id: str | None,
    limit: int,
) -> list[MissingEmbeddingRow]:
    method = getattr(store, "list_missing_embedding_rows", None)
    if not callable(method):
        raise AttributeError("store does not implement list_missing_embedding_rows")
    values = method(
        session_id,
        after_created_at=after_created_at,
        after_id=after_id,
        limit=limit,
    )
    rows = [MissingEmbeddingRow.from_value(value) for value in values]
    return sorted(rows, key=lambda row: (row.created_at, row.id))


def _compare_and_set_batch(
    store: SemanticEnrichmentStore,
    writes: Sequence[SemanticEmbeddingWrite],
) -> int:
    batch_method = getattr(store, "compare_and_set_embeddings", None)
    if callable(batch_method):
        updates = [
            (write.kind, write.id, write.content, list(write.embedding))
            for write in writes
        ]
        result = batch_method(updates)
        if isinstance(result, bool):
            return len(writes) if result else 0
        if isinstance(result, int):
            return result
        if isinstance(result, (set, frozenset)):
            return len(result)
        if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)):
            return len(result)
        raise TypeError("compare_and_set_embeddings returned an invalid result")
    method = getattr(store, "compare_and_set_embedding", None)
    if not callable(method):
        raise AttributeError("store does not implement compare_and_set_embedding")
    written = 0
    for write in writes:
        if method(write.kind, write.id, write.content, list(write.embedding)):
            written += 1
    return written


def _session_active(store: SemanticEnrichmentStore, session_id: str) -> bool:
    method = getattr(store, "is_session_active", None)
    if callable(method):
        return bool(method(session_id))
    getter = getattr(store, "get_session", None)
    if not callable(getter):
        raise AttributeError("store does not implement is_session_active")
    value = getter(session_id)
    if value is None:
        return False
    if isinstance(value, Mapping):
        return not bool(value.get("archived_at"))
    return not bool(getattr(value, "archived_at", None))


def _content_sha256(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if value))


class ThreadSemanticBackend:
    """Single-consumer bounded backend for the in-process job mode."""

    def __init__(
        self,
        handler: Callable[[SemanticEnrichmentRequest], SemanticEnrichmentOutcome],
        max_pending: int = THREAD_PENDING_MAX,
        *,
        name: str = "deepresearch-semantic",
        failure_sink: OutcomeSink | None = None,
    ) -> None:
        self._handler = handler
        self._failure_sink = failure_sink
        self._queue: queue.Queue[SemanticEnrichmentRequest | object] = queue.Queue(
            maxsize=max(1, int(max_pending))
        )
        self._sentinel = object()
        self._pending_sessions: set[str] = set()
        self._running_sessions: set[str] = set()
        self._rerun_requests: dict[str, SemanticEnrichmentRequest] = {}
        self._lock = threading.Lock()
        self._accepting = True
        self._thread = threading.Thread(target=self._consume, name=name, daemon=True)
        self._thread.start()

    def enqueue(self, request: SemanticEnrichmentRequest) -> bool:
        with self._lock:
            if not self._accepting:
                return False
            if request.session_id in self._pending_sessions:
                # A queued/running request may already have passed its final empty
                # scan when a new run commits continuity rows. Remember one latest
                # rerun so coalescing cannot lose that wake-up.
                self._rerun_requests[request.session_id] = request
                return True
            self._pending_sessions.add(request.session_id)
        try:
            self._queue.put_nowait(request)
        except queue.Full:
            with self._lock:
                self._pending_sessions.discard(request.session_id)
            return False
        return True

    def shutdown(self, *, timeout_s: float = 5.0) -> None:
        with self._lock:
            if not self._accepting:
                return
            self._accepting = False
        timeout = max(0.0, float(timeout_s))
        try:
            self._queue.put(self._sentinel, timeout=timeout)
        except queue.Full:
            # The consumer observes ``_accepting=False`` and exits after draining
            # accepted work even when a full queue prevented this bounded sentinel
            # insertion.
            pass
        self._thread.join(timeout=timeout)

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending_sessions)

    def _consume(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is self._sentinel:
                    return
                assert isinstance(item, SemanticEnrichmentRequest)
                request = item
                while True:
                    with self._lock:
                        self._running_sessions.add(request.session_id)
                    try:
                        self._handler(request)
                    except Exception as error:  # noqa: BLE001 - keep the consumer available
                        failure = SemanticEnrichmentOutcome(
                            status="failed",
                            session_id=request.session_id,
                            run_id=request.run_id,
                            trigger=request.trigger,
                            reason_codes=("failed_internal",),
                        )
                        if self._failure_sink is not None:
                            try:
                                self._failure_sink(failure)
                            except Exception:  # noqa: BLE001 - observability is best-effort
                                pass
                        _LOG.warning(
                            "semantic_thread_handler_failed",
                            run_id=request.run_id or None,
                            trigger=request.trigger,
                            error_type=error.__class__.__name__,
                        )
                    with self._lock:
                        rerun = self._rerun_requests.pop(request.session_id, None)
                        if rerun is None:
                            self._running_sessions.discard(request.session_id)
                            self._pending_sessions.discard(request.session_id)
                    if rerun is None:
                        break
                    request = rerun
            finally:
                self._queue.task_done()
            with self._lock:
                stop_after_drain = not self._accepting and self._queue.empty()
            if stop_after_drain:
                return


class RQSemanticEnrichmentBackend:
    """Secret-free enqueue client for the dedicated semantic RQ queue."""

    def __init__(self, connection: Any, *, queue_name: str = SEMANTIC_QUEUE_NAME) -> None:
        try:
            from rq import Queue
        except Exception as error:  # noqa: BLE001
            raise RuntimeError("RQ semantic backend requires the 'rq' package") from error
        self._queue = Queue(queue_name, connection=connection)

    def enqueue(self, request: SemanticEnrichmentRequest) -> str:
        job_id = f"semantic-{uuid.uuid4().hex}"
        self._queue.enqueue(
            "deepresearch_core.baseline.semantic_worker.run_job",
            request.to_payload(),
            job_id=job_id,
            job_timeout=SEMANTIC_JOB_TIMEOUT_S,
            result_ttl=3600,
            failure_ttl=86400,
        )
        return job_id

    def defer(self, request: SemanticEnrichmentRequest, *, delay_s: float = 5.0) -> str:
        job_id = f"semantic-{uuid.uuid4().hex}"
        self._queue.enqueue_in(
            timedelta(seconds=max(1.0, float(delay_s))),
            "deepresearch_core.baseline.semantic_worker.run_job",
            replace(request, attempt=request.attempt + 1).to_payload(),
            job_id=job_id,
            job_timeout=SEMANTIC_JOB_TIMEOUT_S,
            result_ttl=3600,
            failure_ttl=86400,
        )
        return job_id
