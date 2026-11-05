"""RQ worker entrypoint for out-of-process research runs (DR-ITER-021).

In the Docker compose stack the ``worker`` service runs ``python -m
deepresearch_core.baseline.worker``, consuming the ``deepresearch`` queue. Each job
calls :func:`run_job`, which lazily builds (and process-caches) a single
:class:`BaselineService` and executes the run against the shared SQLite DB.

The worker and the API are separate processes sharing one SQLite file; cross-process
write contention is handled by ``PRAGMA busy_timeout`` (see ``store.py``) and cancel is
observed via the persisted ``cancelling`` run status (see ``orchestrator._check_cancel``).
"""

from __future__ import annotations

import json
import os
import sys
import time

# ``python -m deepresearch_core.baseline.worker`` executes this file as
# ``__main__``.  RQ later resolves the dotted job path in every forked work
# horse; without this alias it imports the entire worker module again before
# ``run_job`` can record its start time.  Reusing the already-loaded module
# removes that repeat import while preserving RQ's process isolation and hard
# job timeout behavior.
sys.modules.setdefault("deepresearch_core.baseline.worker", sys.modules[__name__])

from ..observability import configure_logging, get_logger
from .budget import RunRequest, retrieval_budget_for
from .errors import BaselineError, ErrorCode
from .job_backend import redis_url
from .store import Store

_service = None  # cached only inside one process; RQ forks a fresh horse per job
_last_provider_settings = None  # in-memory only; never logged or persisted
_last_provider_sources = None
_last_llm_registry = None  # decrypted in memory only; never logged or persisted


def _get_service():
    """Return a process-wide BaselineService, building it once per worker process."""

    global _service, _last_provider_settings, _last_provider_sources, _last_llm_registry
    if _service is None:
        # build_service() lives at the repo root (copied into the worker image) and
        # already resolves BYOK settings, DB path, and encryption key the same way the
        # API does, guaranteeing identical runtime behavior across processes.
        from run_api import build_service

        _service = build_service()
        # A fresh RQ horse has already wired this exact snapshot. Compare the
        # mandatory per-job reload against it instead of rebuilding it twice.
        _last_provider_settings = _service._provider
        _last_provider_sources = dict(_service._search_provider_sources)
        _last_llm_registry = _service._registry
    return _service


def _open_worker_store() -> Store:
    """Open the shared run store without constructing provider/search runtimes."""

    from run_api import _db_path

    return Store(
        _db_path(),
        encryption_key=str(os.environ.get("DEEPRESEARCH_DB_KEY") or "").strip() or None,
    )


def _json_object(value) -> dict:  # noqa: ANN001
    if isinstance(value, dict):
        return dict(value)
    if not value:
        return {}
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _active_generation(rows) -> str | None:  # noqa: ANN001
    active: str | None = None
    for row in rows:
        payload = _json_object(row["payload"])
        generation_id = str(payload.get("generation_id") or "") or None
        if row["type"] == "answer_delta" and generation_id:
            active = generation_id
        elif row["type"] == "answer_reset" and generation_id == active:
            active = None
        elif row["type"] in {"workflow_end", "error"}:
            active = None
    return active


def _terminalize_expired_run(
    request: RunRequest,
    run_id: str,
    *,
    worker_started_at: float,
) -> bool:
    """Persist a pre-provider timeout/cancel terminal state using only ``Store``."""

    store = _open_worker_store()
    try:
        # API and worker use separate SQLite connections. Reserve the write lock so
        # cancellation and timeout cannot both win the same queued run.
        with store._lock:  # noqa: SLF001 - package-local atomic transition
            connection = store._conn  # noqa: SLF001
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status,metadata FROM runs WHERE id=?",
                (run_id,),
            ).fetchone()
            if row is None or str(row["status"] or "") in {
                "completed",
                "completed_degraded",
                "cancelled",
                "error",
                "partial_timeout",
            }:
                connection.rollback()
                return False

            cancelling = str(row["status"] or "") == "cancelling"
            code = ErrorCode.RUN_CANCELLED if cancelling else ErrorCode.RUN_TIMEOUT
            final_status = "cancelled" if cancelling else "error"
            message = "run was cancelled" if cancelling else "Run timed out before execution started."
            terminal_at = time.time()
            retrieval_budget = retrieval_budget_for(request.mode)
            metadata = _json_object(row["metadata"])
            metadata["retrieval_budget"] = retrieval_budget.to_dict()
            metadata["phase_metrics"] = [
                {
                    "stage": "queue_wait",
                    "attempt": 0,
                    "started_at": request.requested_at or worker_started_at,
                    "finished_at": worker_started_at,
                    "duration_ms": (
                        max(0.0, (worker_started_at - request.requested_at) * 1000.0)
                        if request.requested_at > 0
                        else 0.0
                    ),
                    "status": "cancelled" if cancelling else "timeout",
                    "deadline_remaining_before_ms": (
                        max(0.0, (request.deadline_at - request.requested_at) * 1000.0)
                        if request.deadline_at > 0 and request.requested_at > 0
                        else None
                    ),
                    "deadline_remaining_after_ms": 0.0,
                }
            ]
            metadata["deadline_summary"] = {
                "requested_at": request.requested_at or 0.0,
                "deadline_at": request.deadline_at or 0.0,
                "worker_started_at": worker_started_at,
                "writer_started_at": None,
                "terminal_at": terminal_at,
                "queue_delay_ms": (
                    max(0.0, (worker_started_at - request.requested_at) * 1000.0)
                    if request.requested_at > 0
                    else None
                ),
                "writer_reserve_s": retrieval_budget.writer_reserve_s,
                "finalize_reserve_s": retrieval_budget.finalize_reserve_s,
                "cleanup_reserve_s": retrieval_budget.cleanup_reserve_s,
                "deadline_met": False,
                "cleanup_overrun_ms": (
                    max(0.0, (terminal_at - request.deadline_at) * 1000.0)
                    if request.deadline_at > 0
                    else 0.0
                ),
            }
            error_payload = BaselineError(code, message).to_payload()
            event_rows = connection.execute(
                "SELECT type,payload FROM run_events WHERE run_id=? ORDER BY seq ASC",
                (run_id,),
            ).fetchall()
            active_generation = _active_generation(event_rows)
            next_seq = int(
                connection.execute(
                    "SELECT COALESCE(MAX(seq),0)+1 AS seq FROM run_events WHERE run_id=?",
                    (run_id,),
                ).fetchone()["seq"]
            )
            if active_generation is not None:
                reset_reason = "cancelled" if cancelling else "deadline_exceeded"
                connection.execute(
                    "INSERT INTO run_events(run_id,seq,type,payload,created_at) VALUES (?,?,?,?,?)",
                    (
                        run_id,
                        next_seq,
                        "answer_reset",
                        json.dumps({"generation_id": active_generation, "reason": reset_reason}),
                        terminal_at,
                    ),
                )
                next_seq += 1
            connection.execute(
                "UPDATE runs SET status=?,error=?,metadata=?,updated_at=? WHERE id=?",
                (
                    final_status,
                    json.dumps(error_payload),
                    json.dumps(metadata),
                    terminal_at,
                    run_id,
                ),
            )
            connection.execute(
                "INSERT INTO run_events(run_id,seq,type,payload,created_at) VALUES (?,?,?,?,?)",
                (run_id, next_seq, "error", json.dumps(error_payload), terminal_at),
            )
            connection.commit()
            return True
    except Exception:
        try:
            store._conn.rollback()  # noqa: SLF001
        except Exception:  # noqa: BLE001 - preserve the original terminalization error
            pass
        raise
    finally:
        store.close()


def run_job(payload: dict, run_id: str) -> None:
    """RQ job: execute a previously-persisted run by id.

    ``payload`` round-trips through :meth:`RunRequest.to_payload`; the run row already
    exists (created by the API), so we execute against the existing ``run_id`` rather
    than creating a new run.
    """

    request = RunRequest.from_payload(payload)
    worker_started_at = time.time()
    # ``deadline_at == 0`` is a legacy RQ payload. Preserve its historical
    # worker-start timeout rather than deriving expiry from metrics-only requested_at.
    if request.deadline_at > 0 and worker_started_at >= request.deadline_at:
        _terminalize_expired_run(request, run_id, worker_started_at=worker_started_at)
        return
    global _last_provider_settings, _last_provider_sources, _last_llm_registry
    service = _get_service()
    # RQ workers are process-cached. Reload credentials/readiness before every job,
    # but rebuild provider runtimes only when the in-memory settings actually differ;
    # the run's already-frozen search policy remains unchanged either way.
    loaded_settings = _load_worker_provider_settings()
    llm_registry = None
    if isinstance(loaded_settings, tuple):
        provider_settings, provider_sources = loaded_settings[:2]
        if len(loaded_settings) > 2:
            llm_registry = loaded_settings[2]
    else:  # Backward-compatible test/extension hook.
        provider_settings, provider_sources = loaded_settings, {}
    registry_changed = llm_registry is not None and llm_registry != _last_llm_registry
    if (
        provider_settings != _last_provider_settings
        or provider_sources != _last_provider_sources
        or registry_changed
    ):
        try:
            service.reconfigure(
                provider_settings,
                credential_sources=provider_sources,
                reload_search_provider_state=True,
            )
        except TypeError as error:
            if "unexpected keyword" not in str(error):
                raise
            service.reconfigure(provider_settings)
        _last_provider_settings = provider_settings
        _last_provider_sources = provider_sources
        if llm_registry is not None:
            _last_llm_registry = llm_registry
    elif hasattr(service, "reload_search_provider_state_for_worker"):
        # Probe and enabled state can change without changing credential bytes.
        # Reload that secret-free state on every job while preserving the run's
        # already-frozen chains and avoiding an unnecessary adapter rebuild.
        service.reload_search_provider_state_for_worker()
    if hasattr(service, "reload_post_run_memory_settings_for_worker"):
        # The API and RQ worker are separate, process-cached services.  Observe
        # Settings updates (including a temporary acceptance-test enable) at the
        # start of every run instead of retaining the worker-start snapshot.
        service.reload_post_run_memory_settings_for_worker()
    if hasattr(service, "enforce_worker_search_credential_snapshot"):
        request = service.enforce_worker_search_credential_snapshot(request)
    service.execute_run(request, run_id)


def _load_worker_provider_settings():
    """Reload BYOK provider and LLM registry settings for a cached worker."""

    from run_api import _load_settings_with_sources, default_config_path
    from deepresearch_core.config import load_llm_registry

    loaded = _load_settings_with_sources()
    if isinstance(loaded, tuple):
        provider, sources = loaded[:2]
    else:
        provider, sources = loaded, {}
    registry = load_llm_registry(default_config_path(), environ=os.environ)
    return provider, sources, registry


def run_memory_dream_job(payload: dict, dream_run_id: str) -> None:
    """RQ job: reload local config and execute one persisted Memory Dream."""

    service = _get_service()
    loaded_settings = _load_worker_provider_settings()
    provider = loaded_settings[0] if isinstance(loaded_settings, tuple) else loaded_settings
    service.reload_memory_runtime_for_worker(provider)
    service.execute_memory_dream_job(payload, dream_run_id)


def run_post_turn_memory_job(payload: dict) -> dict:
    """RQ maintenance job: reload config and capture one completed turn."""

    service = _get_service()
    loaded_settings = _load_worker_provider_settings()
    provider = loaded_settings[0] if isinstance(loaded_settings, tuple) else loaded_settings
    service.reload_memory_runtime_for_worker(provider)
    return service.execute_post_turn_memory_job(dict(payload))


def mark_post_turn_memory_failed(job, connection, exc_type=None, exc_value=None, traceback=None) -> None:  # noqa: ANN001, ARG001
    """Persist a secret-free failed audit if RQ kills the maintenance horse."""

    log = get_logger("deepresearch.worker")
    args = list(getattr(job, "args", None) or [])
    payload = dict(args[0]) if args and isinstance(args[0], dict) else {}
    run_id = str(payload.get("run_id") or "")
    session_id = str(payload.get("session_id") or "")
    reason = getattr(exc_type, "__name__", None) or "RQ_JOB_FAILED"
    if not run_id:
        log.warning("post_turn_memory_failure_missing_id", reason=reason)
        return
    try:
        transitioned = _get_service().fail_post_turn_memory_job(
            payload,
            error_code=reason,
        )
        log.warning(
            "post_turn_memory_marked_failed" if transitioned else "post_turn_memory_failure_terminal_noop",
            run_id=run_id,
            session_id=session_id,
            reason=reason,
        )
    except Exception as error:  # noqa: BLE001 - failure callbacks must never raise
        log.warning(
            "mark_post_turn_memory_failed_error",
            run_id=run_id,
            session_id=session_id,
            reason=reason,
            error_type=error.__class__.__name__,
        )
        return


def mark_run_failed(job, connection, exc_type=None, exc_value=None, traceback=None) -> None:  # noqa: ANN001
    """RQ ``on_failure`` callback: persist a terminal error for a killed/crashed job.

    Runs in the worker *main* process (the work horse is already dead), including
    when RQ SIGKILLs the horse on ``job_timeout``. Without this the run row stays
    ``running`` forever and the UI shows an endlessly spinning, message-less session
    (DR-ITER-058). job_id == run_id by construction (see ``RQJobBackend.enqueue``).
    """

    log = get_logger("deepresearch.worker")
    run_id = str(getattr(job, "id", "") or "")
    reason = getattr(exc_type, "__name__", None) or "unknown"
    error_text = f"{reason} {exc_value or ''}".lower()
    is_timeout = "timeout" in error_text or "timed out" in error_text
    code = ErrorCode.RUN_TIMEOUT if is_timeout else ErrorCode.INTERNAL_ERROR
    message = (
        "Run timed out before completion."
        if is_timeout
        else "Run terminated unexpectedly (worker crash)."
    )
    try:
        service = _get_service()
        service.fail_run(run_id, message, code=code)
        log.warning("run_marked_failed", run_id=run_id, reason=reason)
    except Exception:  # noqa: BLE001 - a failure callback must never raise
        log.exception("mark_run_failed_error", run_id=run_id, reason=reason)


def mark_memory_dream_failed(job, connection, exc_type=None, exc_value=None, traceback=None) -> None:  # noqa: ANN001
    """RQ failure callback: close a queued/running Dream without exposing details."""

    log = get_logger("deepresearch.worker")
    args = list(getattr(job, "args", None) or [])
    dream_run_id = str(args[1] if len(args) > 1 else "")
    reason = getattr(exc_type, "__name__", None) or "unknown"
    if not dream_run_id:
        log.warning("memory_dream_failure_missing_id", reason=reason)
        return
    try:
        service = _get_service()
        service.fail_memory_dream(dream_run_id, "Memory Dream worker terminated unexpectedly.")
        log.warning("memory_dream_marked_failed", dream_run_id=dream_run_id, reason=reason)
    except Exception:  # noqa: BLE001 - a failure callback must never raise
        log.exception("mark_memory_dream_failed_error", dream_run_id=dream_run_id, reason=reason)


def main() -> None:
    """Start an RQ worker consuming the DeepResearch queue."""

    configure_logging()
    log = get_logger("deepresearch.worker")
    try:
        import redis as redis_lib
        from rq import Queue, Worker
    except Exception as error:  # noqa: BLE001
        raise SystemExit(
            "The RQ worker requires the 'rq' and 'redis' packages (pip install rq redis)."
        ) from error

    url = redis_url()
    connection = redis_lib.Redis.from_url(url)
    queue = Queue("deepresearch", connection=connection)
    log.info("worker_start", redis_url=url, queue="deepresearch")
    worker = Worker([queue], connection=connection)
    # with_scheduler is unsupported on some Redis builds; degrade gracefully.
    try:
        worker.work(with_scheduler=True)
    except TypeError:
        worker.work()


if __name__ == "__main__":
    main()
