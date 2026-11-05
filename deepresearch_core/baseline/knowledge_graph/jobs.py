"""Dedicated job backend for knowledge graph rebuilds."""

from __future__ import annotations

import threading
from typing import Any, Callable, Protocol

from ..job_backend import job_timeout_s, redis_url, resolve_backend_name


JsonDict = dict[str, Any]
JobFn = Callable[[JsonDict, str], None]
_KG_FAILURE_CALLBACK_PATH = "deepresearch_core.baseline.knowledge_graph.worker.mark_rebuild_failed"
_KG_WORKER_PATH = "deepresearch_core.baseline.knowledge_graph.worker.rebuild_job"
_WIKI_FAILURE_CALLBACK_PATH = "deepresearch_core.baseline.knowledge_graph.wiki_worker.mark_wiki_refresh_failed"
_WIKI_WORKER_PATH = "deepresearch_core.baseline.knowledge_graph.wiki_worker.refresh_wiki_job"


class KnowledgeGraphJobBackend(Protocol):
    """Minimal contract for scheduling KG rebuilds."""

    name: str

    def enqueue(self, payload: JsonDict, job_id: str) -> str:
        """Schedule a rebuild and return the backend job id."""


class ThreadKnowledgeGraphJobBackend:
    """Execute KG rebuilds in in-process daemon threads."""

    name = "thread"

    def __init__(self, job_fn: JobFn) -> None:
        self._job_fn = job_fn
        self._threads: dict[str, threading.Thread] = {}

    def enqueue(self, payload: JsonDict, job_id: str) -> str:
        thread = threading.Thread(target=self._job_fn, args=(payload, job_id), daemon=True)
        self._threads[job_id] = thread
        thread.start()
        return job_id


class RQKnowledgeGraphJobBackend:
    """Enqueue KG rebuilds onto the existing DeepResearch Redis queue."""

    name = "rq"

    def __init__(self, queue: Any, *, failure_callback: Any = None) -> None:
        self._queue = queue
        self._failure_callback = failure_callback

    def enqueue(self, payload: JsonDict, job_id: str) -> str:
        backend_job_id = job_id
        kwargs: JsonDict = {
            "job_id": backend_job_id,
            "result_ttl": 3600,
            "failure_ttl": 86400,
            "job_timeout": job_timeout_s(),
        }
        if self._failure_callback is not None:
            kwargs["on_failure"] = self._failure_callback
        self._queue.enqueue(_KG_WORKER_PATH, payload, job_id, **kwargs)
        return backend_job_id


class RQWikiMapJobBackend:
    """Enqueue Wiki Map refreshes onto the existing DeepResearch Redis queue."""

    name = "rq"

    def __init__(self, queue: Any, *, failure_callback: Any = None) -> None:
        self._queue = queue
        self._failure_callback = failure_callback

    def enqueue(self, payload: JsonDict, job_id: str) -> str:
        backend_job_id = job_id
        kwargs: JsonDict = {
            "job_id": backend_job_id,
            "result_ttl": 3600,
            "failure_ttl": 86400,
            "job_timeout": job_timeout_s(),
        }
        if self._failure_callback is not None:
            kwargs["on_failure"] = self._failure_callback
        self._queue.enqueue(_WIKI_WORKER_PATH, payload, job_id, **kwargs)
        return backend_job_id


def create_knowledge_graph_job_backend(*, job_fn: JobFn) -> KnowledgeGraphJobBackend:
    """Construct the backend selected by the same env knob as research jobs."""

    if resolve_backend_name() != "rq":
        return ThreadKnowledgeGraphJobBackend(job_fn)
    try:
        import redis as redis_lib
        from rq import Callback, Queue
    except Exception as error:  # noqa: BLE001
        from ..job_backend import JobBackendError

        raise JobBackendError(
            "DEEPRESEARCH_JOB_BACKEND=rq requires the 'rq' and 'redis' packages "
            "(pip install rq redis)."
        ) from error
    connection = redis_lib.Redis.from_url(redis_url(), socket_connect_timeout=2)
    connection.ping()
    try:
        failure_callback = Callback(_KG_FAILURE_CALLBACK_PATH)
    except Exception:  # noqa: BLE001
        failure_callback = None
    return RQKnowledgeGraphJobBackend(Queue("deepresearch", connection=connection), failure_callback=failure_callback)


def create_wiki_map_job_backend(*, job_fn: JobFn) -> KnowledgeGraphJobBackend:
    """Construct the backend selected by the same env knob for Wiki Map refreshes."""

    if resolve_backend_name() != "rq":
        return ThreadKnowledgeGraphJobBackend(job_fn)
    try:
        import redis as redis_lib
        from rq import Callback, Queue
    except Exception as error:  # noqa: BLE001
        from ..job_backend import JobBackendError

        raise JobBackendError(
            "DEEPRESEARCH_JOB_BACKEND=rq requires the 'rq' and 'redis' packages "
            "(pip install rq redis)."
        ) from error
    connection = redis_lib.Redis.from_url(redis_url(), socket_connect_timeout=2)
    connection.ping()
    try:
        failure_callback = Callback(_WIKI_FAILURE_CALLBACK_PATH)
    except Exception:  # noqa: BLE001
        failure_callback = None
    return RQWikiMapJobBackend(Queue("deepresearch", connection=connection), failure_callback=failure_callback)
