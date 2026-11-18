"""Desktop-only recovery and bounded shutdown for in-process workers."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Iterable


_ACTIVE_RUN_STATUSES = ("queued", "running", "pausing", "cancelling")
_ACTIVE_JOB_STATUSES = ("queued", "running")
_INTERRUPTED_MESSAGE = "The previous desktop process ended before this task completed."
_INTERRUPTED_CODE = "DESKTOP_PROCESS_INTERRUPTED"


@dataclass(frozen=True)
class DesktopRecoveryResult:
    runs: int = 0
    memory_dreams: int = 0
    post_turn_memory: int = 0
    graph_jobs: int = 0
    wiki_jobs: int = 0


@dataclass(frozen=True)
class DesktopShutdownResult:
    safe_to_close_store: bool
    pending_threads: tuple[str, ...]
    recovery: DesktopRecoveryResult | None = None


def recover_interrupted_desktop_state(service: Any) -> DesktopRecoveryResult:
    """Terminalize records whose owning desktop process no longer exists.

    Call this after acquiring the desktop's exclusive loopback listener. Successful
    and intentionally paused work is outside every reconciliation query.
    """

    store = service.store
    run_count = 0
    for row in store.list_runs_by_statuses(_ACTIVE_RUN_STATUSES):
        run_id = str(row.get("id") or "")
        if not run_id:
            continue
        service.cancel_run(run_id)
        service.fail_run(run_id, _INTERRUPTED_MESSAGE)
        if str((store.get_run(run_id) or {}).get("status") or "") == "error":
            run_count += 1

    dream_count = 0
    for row in store.list_active_memory_dream_runs():
        dream_id = str(row.get("id") or "")
        if not dream_id:
            continue
        service.fail_memory_dream(
            dream_id,
            _INTERRUPTED_MESSAGE,
            error_code=_INTERRUPTED_CODE,
        )
        dream_count += 1

    post_turn_count = 0
    for row in store.list_active_post_turn_memory_runs():
        changed = store.fail_post_turn_memory_run(
            run_id=str(row.get("run_id") or ""),
            session_id=str(row.get("session_id") or ""),
            error_code=_INTERRUPTED_CODE,
        )
        post_turn_count += int(changed)

    graph_count = 0
    wiki_count = 0
    for job_id in _active_job_ids(store, "kg_build_jobs"):
        service.mark_knowledge_graph_rebuild_failed(job_id, message=_INTERRUPTED_MESSAGE)
        graph_count += 1
    for job_id in _active_job_ids(store, "kg_wiki_refresh_jobs"):
        service.mark_knowledge_graph_wiki_refresh_failed(job_id, message=_INTERRUPTED_MESSAGE)
        wiki_count += 1

    return DesktopRecoveryResult(
        runs=run_count,
        memory_dreams=dream_count,
        post_turn_memory=post_turn_count,
        graph_jobs=graph_count,
        wiki_jobs=wiki_count,
    )


def shutdown_desktop_service(service: Any, *, timeout_s: float = 5.0) -> DesktopShutdownResult:
    """Cancel work and wait a bounded time after the API has stopped accepting requests.

    The caller may explicitly close ``service.store`` only when the returned
    ``safe_to_close_store`` flag is true.
    """

    deadline = time.monotonic() + max(0.0, float(timeout_s))
    store = service.store
    for row in store.list_runs_by_statuses(_ACTIVE_RUN_STATUSES):
        run_id = str(row.get("id") or "")
        if run_id:
            service.cancel_run(run_id)
    for row in store.list_active_memory_dream_runs():
        dream_id = str(row.get("id") or "")
        if dream_id:
            service.cancel_memory_dream(dream_id)

    semantic_threads = _semantic_threads(service)
    shutdown_thread = threading.Thread(
        target=service.shutdown,
        name="desktop-service-shutdown",
        daemon=True,
    )
    shutdown_thread.start()
    pending = _join_owned_threads(
        service,
        deadline=deadline,
        extra=(("service-shutdown", shutdown_thread), *semantic_threads),
    )
    if pending:
        return DesktopShutdownResult(False, pending)

    recovery = recover_interrupted_desktop_state(service)
    return DesktopShutdownResult(True, (), recovery)


def _active_job_ids(store: Any, table: str) -> tuple[str, ...]:
    if table not in {"kg_build_jobs", "kg_wiki_refresh_jobs"}:
        raise ValueError("unsupported desktop job table")
    placeholders = ",".join("?" for _ in _ACTIVE_JOB_STATUSES)
    with store._lock:  # noqa: SLF001 - the desktop adapter owns lifecycle reconciliation
        rows = store._conn.execute(  # noqa: SLF001
            f"SELECT id FROM {table} WHERE status IN ({placeholders}) ORDER BY started_at, id",
            _ACTIVE_JOB_STATUSES,
        ).fetchall()
    return tuple(str(row["id"]) for row in rows)


def _join_owned_threads(
    service: Any,
    *,
    deadline: float,
    extra: tuple[tuple[str, threading.Thread], ...] = (),
) -> tuple[str, ...]:
    """Join snapshots until no producer can publish another owned worker."""

    while True:
        alive = [*extra, *_owned_threads(service)]
        alive = [(name, thread) for name, thread in alive if thread.is_alive()]
        if not alive:
            return ()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return tuple(sorted(name for name, _thread in alive))
        slice_s = remaining / max(1, len(alive))
        for _name, thread in alive:
            thread.join(max(0.0, slice_s))


def _owned_threads(service: Any) -> list[tuple[str, threading.Thread]]:
    groups: list[tuple[str, Iterable[threading.Thread]]] = [
        ("research", _mapping_threads(getattr(service, "_job_backend", None))),
        (
            "graph",
            _mapping_threads(getattr(getattr(service, "_knowledge_graph", None), "_job_backend", None)),
        ),
        ("wiki", _mapping_threads(getattr(service, "_wiki_job_backend", None))),
        ("memory-refresh", _sequence_threads(getattr(service, "_memory_refresh_queue", None))),
        ("graph-refresh", _sequence_threads(getattr(service, "_knowledge_graph_refresh_queue", None))),
    ]
    alive: list[tuple[str, threading.Thread]] = []
    for prefix, threads in groups:
        for index, thread in enumerate(threads):
            if thread.is_alive():
                alive.append((f"{prefix}:{thread.name or index}", thread))
    return alive


def _mapping_threads(owner: Any) -> tuple[threading.Thread, ...]:
    threads = getattr(owner, "_threads", {})
    return tuple(threads.values()) if isinstance(threads, dict) else ()


def _sequence_threads(owner: Any) -> tuple[threading.Thread, ...]:
    threads = getattr(owner, "_threads", ())
    return tuple(threads) if isinstance(threads, (list, tuple)) else ()


def _semantic_threads(service: Any) -> tuple[tuple[str, threading.Thread], ...]:
    backend = getattr(service, "_semantic_backend", None)
    thread = getattr(backend, "_thread", None)
    if isinstance(thread, threading.Thread):
        return ((f"semantic:{thread.name}", thread),)
    return ()
