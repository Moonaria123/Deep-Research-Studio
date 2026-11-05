"""Dedicated RQ worker for deadline-independent semantic maintenance jobs."""

from __future__ import annotations

import os
import sys
from typing import Any, Mapping

sys.modules.setdefault("deepresearch_core.baseline.semantic_worker", sys.modules[__name__])

from ..observability import configure_logging, get_logger
from .job_backend import redis_url
from .semantic_enrichment import (
    RESEARCH_QUEUE_NAME,
    SEMANTIC_JOB_TIMEOUT_S,
    SEMANTIC_QUEUE_NAME,
    RQSemanticEnrichmentBackend,
    SemanticEnrichmentOutcome,
    SemanticEnrichmentRequest,
)


_service = None
_MAX_QUEUE_DEFERRALS = 10


def _get_service():
    """Build the semantic job service from the same authorized config as API/worker."""

    global _service
    if _service is None:
        from run_api import build_service

        _service = build_service()
    return _service


def _load_provider_settings():
    from run_api import _load_settings_with_sources

    return _load_settings_with_sources()


def _redis_connection():
    try:
        import redis as redis_lib
    except Exception as error:  # noqa: BLE001
        raise RuntimeError("Semantic RQ jobs require the 'redis' package") from error
    return redis_lib.Redis.from_url(
        redis_url(),
        socket_connect_timeout=2,
        socket_timeout=2,
        health_check_interval=30,
    )


def run_job(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one semantic maintenance request synchronously in the RQ work horse."""

    request = SemanticEnrichmentRequest.from_payload(payload)
    connection = _redis_connection()
    if _research_queue_busy(connection):
        return _defer(
            connection,
            request,
            status="deferred",
            reason="deferred_research_busy",
        )

    lock = connection.lock(
        f"deepresearch:semantic:session:{request.session_id}",
        timeout=SEMANTIC_JOB_TIMEOUT_S,
        blocking_timeout=0,
    )
    acquired = bool(lock.acquire(blocking=False))
    if not acquired:
        return _defer(
            connection,
            request,
            status="deferred",
            reason="deferred_session_lease_busy",
        )

    try:
        if _research_queue_busy(connection):
            return _defer(
                connection,
                request,
                status="deferred",
                reason="deferred_research_busy",
            )
        try:
            service = _get_service()
            loaded = _load_provider_settings()
            provider_settings = loaded[0] if isinstance(loaded, tuple) else loaded
            reload_runtime = getattr(service, "reload_semantic_runtime_for_worker", None)
            if callable(reload_runtime):
                reload_runtime(provider_settings)
            execute = getattr(service, "execute_semantic_enrichment_job", None)
            if not callable(execute):
                raise RuntimeError(
                    "service does not implement execute_semantic_enrichment_job"
                )
            result = execute(
                request.to_payload(),
                research_busy=lambda: _research_queue_busy(connection),
            )
            outcome = _outcome_dict(result)
        except Exception:  # noqa: BLE001 - preserve a secret-free terminal outcome
            outcome = SemanticEnrichmentOutcome(
                status="failed",
                session_id=request.session_id,
                run_id=request.run_id,
                trigger=request.trigger,
                reason_codes=("failed_internal",),
            ).to_dict()
        _persist_current_job_meta(outcome)
        return outcome
    finally:
        try:
            lock.release()
        except Exception:  # noqa: BLE001 - an expired lease needs no further action
            pass


def _research_queue_busy(connection: Any) -> bool:
    try:
        from rq import Queue
    except Exception as error:  # noqa: BLE001
        raise RuntimeError("Semantic RQ jobs require the 'rq' package") from error
    return int(Queue(RESEARCH_QUEUE_NAME, connection=connection).count or 0) > 0


def _defer(
    connection: Any,
    request: SemanticEnrichmentRequest,
    *,
    status: str,
    reason: str,
) -> dict[str, Any]:
    outcome = SemanticEnrichmentOutcome(
        status=status,
        session_id=request.session_id,
        run_id=request.run_id,
        trigger=request.trigger,
        reason_codes=(reason,),
    ).to_dict()
    if request.attempt < _MAX_QUEUE_DEFERRALS:
        RQSemanticEnrichmentBackend(connection).defer(request)
        outcome["reason_codes"].append("retry_scheduled")
    else:
        outcome["reason_codes"].append("queue_deferral_limit_reached")
    _persist_current_job_meta(outcome)
    return outcome


def _outcome_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, SemanticEnrichmentOutcome):
        return value.to_dict()
    if isinstance(value, Mapping):
        return dict(value)
    raise RuntimeError("semantic enrichment job returned an invalid outcome")


def _persist_current_job_meta(outcome: Mapping[str, Any]) -> None:
    try:
        from rq import get_current_job

        job = get_current_job()
        if job is None:
            return
        job.meta["semantic_outcome"] = dict(outcome)
        job.save_meta()
    except Exception:  # noqa: BLE001 - outcome metadata is best-effort
        return


def main() -> None:
    """Start a single RQ worker consuming only ``deepresearch-semantic``."""

    configure_logging()
    log = get_logger("deepresearch.semantic_worker")
    try:
        import redis as redis_lib
        from rq import Queue, Worker
    except Exception as error:  # noqa: BLE001
        raise SystemExit(
            "The semantic worker requires the 'rq' and 'redis' packages (pip install rq redis)."
        ) from error

    # RQ owns a long blocking dequeue on this parent-worker connection; applying
    # the short maintenance command timeout here would make an idle worker flap.
    # Each work horse creates its own bounded connection in ``_redis_connection``.
    connection = redis_lib.Redis.from_url(redis_url())
    queue = Queue(SEMANTIC_QUEUE_NAME, connection=connection)
    log.info("semantic_worker_start", queue=SEMANTIC_QUEUE_NAME, pid=os.getpid())
    worker = Worker([queue], connection=connection)
    try:
        worker.work(with_scheduler=True)
    except TypeError:
        worker.work()


if __name__ == "__main__":
    main()
