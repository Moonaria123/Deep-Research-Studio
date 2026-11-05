"""Best-effort post-run Knowledge Graph refresh jobs."""

from __future__ import annotations

import threading
from collections.abc import Callable

from ...observability import get_logger

_LOG = get_logger(__name__)


class KnowledgeGraphRefreshQueue:
    """Run KG refresh triggers inline for tests or in daemon threads for runtime."""

    def __init__(
        self,
        *,
        refresh_fn: Callable[..., None],
        run_inline: bool = False,
    ) -> None:
        self._refresh_fn = refresh_fn
        self._run_inline = run_inline
        self._workspace_locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def enqueue(self, *, workspace_id: str, run_id: str) -> bool:
        """Schedule a KG refresh trigger; returns whether it was accepted."""

        if not workspace_id or not run_id:
            return False
        kwargs = {"workspace_id": workspace_id or "default", "run_id": run_id}
        if self._run_inline:
            self._run(**kwargs)
            return True
        thread = threading.Thread(target=self._run, kwargs=kwargs, daemon=True)
        with self._lock:
            self._threads.append(thread)
        thread.start()
        return True

    def _run(self, *, workspace_id: str, run_id: str) -> None:
        with self._workspace_lock(workspace_id):
            try:
                self._refresh_fn(workspace_id=workspace_id, run_id=run_id)
            except Exception as error:  # noqa: BLE001 - KG refresh must never affect run state
                _LOG.warning(
                    "knowledge_graph_refresh_failed",
                    run_id=run_id,
                    workspace_id=workspace_id,
                    error=error.__class__.__name__,
                )

    def _workspace_lock(self, workspace_id: str) -> threading.Lock:
        with self._lock:
            lock = self._workspace_locks.get(workspace_id)
            if lock is None:
                lock = threading.Lock()
                self._workspace_locks[workspace_id] = lock
            return lock
