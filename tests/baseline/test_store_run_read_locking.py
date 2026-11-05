"""Thread-safety regression tests for run and SSE read paths."""

from __future__ import annotations

import threading
from typing import Any

from deepresearch_core.baseline.store import Store


class _TrackingLock:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.depth = 0

    def __enter__(self) -> "_TrackingLock":
        self._lock.acquire()
        self.depth += 1
        return self

    def __exit__(self, *args: object) -> None:
        self.depth -= 1
        self._lock.release()


class _LockCheckingConnection:
    def __init__(self, connection: Any, lock: _TrackingLock) -> None:
        self._connection = connection
        self._lock = lock

    def execute(self, *args: object, **kwargs: object) -> Any:
        assert self._lock.depth > 0, "SQLite run read executed without Store._lock"
        return self._connection.execute(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._connection, name)


def test_run_and_event_reads_hold_store_lock() -> None:
    store = Store(":memory:")
    session_id = store.create_session(title="thread-safe reads")
    run_id = store.create_run(session_id=session_id, mode="fast", query="q")
    store.append_event(run_id, "workflow_start", {"ok": True})

    tracking_lock = _TrackingLock()
    store._lock = tracking_lock
    store._conn = _LockCheckingConnection(store._conn, tracking_lock)

    assert store.get_latest_run_for_session(session_id) == (run_id, "fast")
    assert store.get_latest_run_id(session_id) == run_id
    assert store.get_run(run_id)["status"] == "queued"
    assert store.list_events(run_id)[0]["type"] == "workflow_start"
    store.close()
