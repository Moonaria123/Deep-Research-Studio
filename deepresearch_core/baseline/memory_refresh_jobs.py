"""Best-effort post-run memory refresh jobs."""

from __future__ import annotations

import threading
from collections.abc import Callable

from ..observability import get_logger

_LOG = get_logger(__name__)


class MemoryRefreshQueue:
    """Run memory refresh jobs inline for tests or in daemon threads for runtime."""

    def __init__(
        self,
        *,
        refresh_fn: Callable[..., None],
        run_inline: bool = False,
    ) -> None:
        self._refresh_fn = refresh_fn
        self._run_inline = run_inline
        self._session_locks: dict[str, threading.Lock] = {}
        self._lock = threading.Lock()
        self._threads: list[threading.Thread] = []

    def enqueue(self, *, session_id: str, run_id: str, workspace_key: str, user_key: str) -> bool:
        """Schedule a refresh for one session; returns whether it was accepted."""

        if not session_id or not run_id:
            return False
        kwargs = {
            "session_id": session_id,
            "run_id": run_id,
            "workspace_key": workspace_key or "default",
            "user_key": user_key or "default",
        }
        if self._run_inline:
            self._run(**kwargs)
            return True
        thread = threading.Thread(target=self._run, kwargs=kwargs, daemon=True)
        with self._lock:
            self._threads.append(thread)
        thread.start()
        return True

    def _run(self, *, session_id: str, run_id: str, workspace_key: str, user_key: str) -> None:
        with self._session_lock(session_id):
            try:
                self._refresh_fn(
                    session_id=session_id,
                    run_id=run_id,
                    workspace_key=workspace_key,
                    user_key=user_key,
                )
            except Exception as error:  # noqa: BLE001 - refresh must never affect run terminal state
                _LOG.warning(
                    "memory_refresh_failed",
                    run_id=run_id,
                    session_id=session_id,
                    error=error.__class__.__name__,
                )

    def _session_lock(self, session_id: str) -> threading.Lock:
        with self._lock:
            lock = self._session_locks.get(session_id)
            if lock is None:
                lock = threading.Lock()
                self._session_locks[session_id] = lock
            return lock
