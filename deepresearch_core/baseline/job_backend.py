"""Pluggable job backend for executing research runs (DR-ITER-021).

Two backends share one HTTP/SSE contract so clients never observe which is active:

* ``thread`` -- run in an in-process daemon thread (default for local/CLI/desktop
  sidecars and the pytest suite; no external dependency).
* ``rq``     -- enqueue to Redis Queue so a separate worker process executes the run,
  surviving API restarts (default for the Docker compose stack).

Selection is environment-driven (``DEEPRESEARCH_JOB_BACKEND``); there is deliberately
no UI/runtime switch. When ``rq`` is requested but Redis/rq is unavailable, backend
construction raises :class:`JobBackendError` so the API fails fast at startup instead
of silently dropping jobs. The ``thread`` path never imports or contacts Redis.
"""

from __future__ import annotations

import math
import os
import threading
from typing import Any, Callable, Protocol

from .budget import RunRequest, retrieval_budget_for, validate_deadline_budget

JsonDict = dict[str, Any]

# Callable signature shared by both backends for executing a run in-process.
RunFn = Callable[[RunRequest, str], None]
MemoryDreamFn = Callable[[JsonDict, str], None]

DEFAULT_REDIS_URL = "redis://redis:6379/0"
_QUEUE_NAME = "deepresearch"
_MAINTENANCE_QUEUE_NAME = "deepresearch-semantic"
# RQ's default job timeout is 180s, which silently kills long Deep Research /
# Council runs ("killed horse"). 1800s is a last-resort guard against a truly
# stuck LLM call, not a normal completion bound (DR-ITER-058).
DEFAULT_JOB_TIMEOUT_S = 1800
_FAILURE_CALLBACK_PATH = "deepresearch_core.baseline.worker.mark_run_failed"
_MEMORY_DREAM_FAILURE_CALLBACK_PATH = "deepresearch_core.baseline.worker.mark_memory_dream_failed"
_POST_TURN_MEMORY_FAILURE_CALLBACK_PATH = "deepresearch_core.baseline.worker.mark_post_turn_memory_failed"


class JobBackendError(RuntimeError):
    """Raised when a requested job backend cannot be constructed (fail-fast)."""


class JobEnqueueStateUnknown(JobBackendError):
    """Raised when Redis may have accepted a job but acknowledgement was lost."""


def resolve_backend_name() -> str:
    """Return the requested backend name from the environment.

    Defaults to ``thread`` so importing/constructing a service never requires Redis;
    the Docker compose stack opts into ``rq`` explicitly via ``DEEPRESEARCH_JOB_BACKEND``.
    """

    value = str(os.environ.get("DEEPRESEARCH_JOB_BACKEND") or "thread").strip().lower()
    return value if value in {"thread", "rq"} else "thread"


def redis_url() -> str:
    """Return the configured Redis URL (compose default points at the ``redis`` service)."""

    return str(os.environ.get("DEEPRESEARCH_REDIS_URL") or DEFAULT_REDIS_URL).strip()


def job_timeout_s() -> int:
    """Return the RQ job timeout in seconds (env ``DEEPRESEARCH_JOB_TIMEOUT_S``).

    Invalid or non-positive values fall back to :data:`DEFAULT_JOB_TIMEOUT_S` so a
    typo can never reinstate RQ's 180s default and start killing healthy runs.
    """

    raw = str(os.environ.get("DEEPRESEARCH_JOB_TIMEOUT_S") or "").strip()
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_JOB_TIMEOUT_S
    return value if value > 0 else DEFAULT_JOB_TIMEOUT_S


def research_job_timeout_s(request: RunRequest) -> int:
    """Return the RQ hard timeout for one research run plus cleanup reserve.

    RQ starts this timer only after dequeue. The run's absolute ``deadline_at`` is
    still authoritative and includes queue delay; this hard timeout is the last-resort
    guard that lets the work horse persist bounded cleanup before RQ kills it.
    """

    budget = request.resolved_budget()
    retrieval_budget = retrieval_budget_for(request.mode)
    validate_deadline_budget(budget, retrieval_budget)
    # Verification may continue after the report's own deadline and terminal event.
    return max(1, int(math.ceil(float(budget.timeout_s) + retrieval_budget.cleanup_reserve_s + 25)))


class JobBackend(Protocol):
    """Minimal contract the service depends on."""

    name: str

    def enqueue(self, request: RunRequest, run_id: str) -> None:
        """Schedule ``request`` (already persisted as ``run_id``) for execution."""

    def health(self) -> JsonDict:
        """Return ``{"job_backend": name, "redis_reachable": bool | None}``."""

    def enqueue_memory_dream(self, payload: JsonDict, dream_run_id: str) -> str:
        """Schedule a persisted Memory Dream and return its backend job id."""

    def enqueue_post_turn_memory(self, payload: JsonDict) -> str:
        """Schedule one post-turn Memory refresh after report publication."""

    def memory_dream_job_status(self, backend_job_id: str) -> str | None:
        """Return the normalized backend status, or ``None`` when missing."""

    def cancel_memory_dream(self, backend_job_id: str) -> str:
        """Request cancellation and return a normalized backend outcome."""


class ThreadJobBackend:
    """Execute runs in in-process daemon threads (the pre-021 behavior)."""

    name = "thread"

    def __init__(self, run_fn: RunFn, *, memory_dream_fn: MemoryDreamFn | None = None) -> None:
        self._run_fn = run_fn
        self._memory_dream_fn = memory_dream_fn
        self._threads: dict[str, threading.Thread] = {}

    def enqueue(self, request: RunRequest, run_id: str) -> None:
        thread = threading.Thread(target=self._run_fn, args=(request, run_id), daemon=True)
        self._threads[run_id] = thread
        thread.start()

    def health(self) -> JsonDict:
        # Redis is irrelevant to the thread backend; report None to signal "n/a".
        return {"job_backend": self.name, "redis_reachable": None}

    def enqueue_memory_dream(self, payload: JsonDict, dream_run_id: str) -> str:
        if self._memory_dream_fn is None:
            raise JobBackendError("thread backend has no Memory Dream worker")
        backend_job_id = f"memory-dream-{dream_run_id}"
        thread = threading.Thread(
            target=self._memory_dream_fn,
            args=(dict(payload), dream_run_id),
            daemon=True,
        )
        self._threads[backend_job_id] = thread
        thread.start()
        return backend_job_id

    def enqueue_post_turn_memory(self, payload: JsonDict) -> str:  # noqa: ARG002
        raise JobBackendError("thread backend uses the in-process Memory refresh queue")

    def memory_dream_job_status(self, backend_job_id: str) -> str | None:
        thread = self._threads.get(backend_job_id)
        if thread is None:
            return None
        return "started" if thread.is_alive() else "finished"

    def cancel_memory_dream(self, backend_job_id: str) -> str:  # noqa: ARG002
        # Python threads cannot be killed safely. The durable cancelling flag is
        # observed by the Memory distiller between bounded stages.
        return "cancelling"


class RQJobBackend:
    """Enqueue runs onto Redis Queue for an out-of-process worker."""

    name = "rq"

    def __init__(
        self,
        queue: Any,
        connection: Any,
        url: str,
        *,
        failure_callback: Any = None,
        memory_dream_failure_callback: Any = None,
        post_turn_memory_failure_callback: Any = None,
        maintenance_queue: Any = None,
    ) -> None:
        self._queue = queue
        self._conn = connection
        self._url = url
        self._failure_callback = failure_callback
        self._memory_dream_failure_callback = memory_dream_failure_callback
        self._post_turn_memory_failure_callback = post_turn_memory_failure_callback
        self._maintenance_queue = maintenance_queue or queue

    def enqueue(self, request: RunRequest, run_id: str) -> None:
        # The worker entrypoint is referenced by dotted path so RQ can import it in the
        # worker process without pickling this backend. job_id == run_id keeps the RQ
        # registry aligned with our run ids (and dedupes accidental double-enqueue).
        kwargs: JsonDict = {
            "job_id": run_id,
            "result_ttl": 3600,
            "failure_ttl": 86400,
            # Research uses its resolved run deadline plus bounded cleanup. Dream/KG
            # retain the independent maintenance timeout returned by job_timeout_s().
            "job_timeout": research_job_timeout_s(request),
        }
        if self._failure_callback is not None:
            # Marks the run as errored (terminal event included) when the job dies
            # for any reason RQ can observe: timeout, OOM-killed horse, crash.
            kwargs["on_failure"] = self._failure_callback
        try:
            self._queue.enqueue(
                "deepresearch_core.baseline.worker.run_job",
                request.to_payload(),
                run_id,
                **kwargs,
            )
        except Exception as error:
            try:
                accepted = self._queue.fetch_job(run_id) is not None
            except Exception as confirm_error:
                raise JobEnqueueStateUnknown(
                    "research queue acceptance could not be confirmed"
                ) from confirm_error
            if accepted:
                return
            raise error

    def health(self) -> JsonDict:
        reachable = False
        try:
            reachable = bool(self._conn.ping())
        except Exception:  # noqa: BLE001 - health must never raise
            reachable = False
        return {"job_backend": self.name, "redis_reachable": reachable}

    def enqueue_memory_dream(self, payload: JsonDict, dream_run_id: str) -> str:
        backend_job_id = f"memory-dream-{dream_run_id}"
        kwargs: JsonDict = {
            "job_id": backend_job_id,
            "result_ttl": 3600,
            "failure_ttl": 86400,
            "job_timeout": job_timeout_s(),
        }
        if self._memory_dream_failure_callback is not None:
            kwargs["on_failure"] = self._memory_dream_failure_callback
        self._queue.enqueue(
            "deepresearch_core.baseline.worker.run_memory_dream_job",
            dict(payload),
            dream_run_id,
            **kwargs,
        )
        return backend_job_id

    def enqueue_post_turn_memory(self, payload: JsonDict) -> str:
        """Persist post-turn Memory work on the deadline-independent queue."""

        run_id = str(payload.get("run_id") or "").strip()
        if not run_id:
            raise JobBackendError("post-turn Memory refresh requires a run id")
        backend_job_id = f"post-turn-memory-{run_id}"
        kwargs: JsonDict = {
            "job_id": backend_job_id,
            "result_ttl": 3600,
            "failure_ttl": 86400,
            "job_timeout": job_timeout_s(),
        }
        if self._post_turn_memory_failure_callback is not None:
            kwargs["on_failure"] = self._post_turn_memory_failure_callback
        self._maintenance_queue.enqueue(
            "deepresearch_core.baseline.worker.run_post_turn_memory_job",
            dict(payload),
            **kwargs,
        )
        return backend_job_id

    def memory_dream_job_status(self, backend_job_id: str) -> str | None:
        try:
            job = self._queue.fetch_job(backend_job_id)
            if job is None:
                return None
            status = job.get_status(refresh=True)
        except Exception:  # noqa: BLE001 - an outage is not proof that the job vanished
            return "unknown"
        value = getattr(status, "value", status)
        return str(value or "").strip().lower() or None

    def cancel_memory_dream(self, backend_job_id: str) -> str:
        job = self._queue.fetch_job(backend_job_id)
        if job is None:
            return "missing"
        status = self.memory_dream_job_status(backend_job_id)
        if status in {"queued", "deferred", "scheduled"}:
            job.cancel()
            return "cancelled"
        if status in {"started", "running"}:
            from rq.command import send_stop_job_command

            send_stop_job_command(self._conn, backend_job_id)
            return "cancelling"
        if status in {"canceled", "cancelled", "stopped"}:
            return "cancelled"
        return status or "missing"


def create_rq_backend(url: str | None = None) -> RQJobBackend:
    """Build an RQ backend, failing fast if rq/redis is missing or unreachable."""

    target = url or redis_url()
    try:
        import redis as redis_lib
        from rq import Queue
    except Exception as error:  # noqa: BLE001
        raise JobBackendError(
            "DEEPRESEARCH_JOB_BACKEND=rq requires the 'rq' and 'redis' packages "
            "(pip install rq redis)."
        ) from error
    # Bound connect time so a misconfigured Redis fails fast at startup and never
    # hangs the /health probe.
    connection = redis_lib.Redis.from_url(target, socket_connect_timeout=2)
    try:
        connection.ping()
    except Exception as error:  # noqa: BLE001
        raise JobBackendError(
            f"DEEPRESEARCH_JOB_BACKEND=rq but Redis is unreachable at {target}. "
            "Start the redis service or set DEEPRESEARCH_JOB_BACKEND=thread."
        ) from error
    queue = Queue(_QUEUE_NAME, connection=connection)
    maintenance_queue = Queue(_MAINTENANCE_QUEUE_NAME, connection=connection)
    # Callback by dotted path: the API process never imports the worker module; RQ
    # resolves it inside the worker process where the failure is handled.
    try:
        from rq import Callback

        failure_callback = Callback(_FAILURE_CALLBACK_PATH)
        memory_dream_failure_callback = Callback(_MEMORY_DREAM_FAILURE_CALLBACK_PATH)
        post_turn_memory_failure_callback = Callback(_POST_TURN_MEMORY_FAILURE_CALLBACK_PATH)
    except Exception:  # noqa: BLE001 - very old rq without Callback support
        failure_callback = None
        memory_dream_failure_callback = None
        post_turn_memory_failure_callback = None
    return RQJobBackend(
        queue,
        connection,
        target,
        failure_callback=failure_callback,
        memory_dream_failure_callback=memory_dream_failure_callback,
        post_turn_memory_failure_callback=post_turn_memory_failure_callback,
        maintenance_queue=maintenance_queue,
    )


def create_job_backend(*, run_fn: RunFn, memory_dream_fn: MemoryDreamFn | None = None) -> JobBackend:
    """Construct the job backend selected by the environment.

    ``thread`` (default) wraps ``run_fn``; ``rq`` builds a Redis-backed queue and fails
    fast when Redis is not available so misconfiguration surfaces at startup.
    """

    if resolve_backend_name() == "rq":
        return create_rq_backend()
    return ThreadJobBackend(run_fn, memory_dream_fn=memory_dream_fn)
