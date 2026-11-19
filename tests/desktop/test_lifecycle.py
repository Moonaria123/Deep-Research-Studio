"""Desktop shutdown and interrupted-process recovery contracts."""

from __future__ import annotations

from types import SimpleNamespace
import threading
import time

from deepresearch_core.baseline.store import Store
from deepresearch_core.baseline import BaselineService
from deepresearch_core.settings import ProviderSettings
from deepresearch_core.desktop_lifecycle import (
    recover_interrupted_desktop_state,
    shutdown_desktop_service,
)


class FakeDesktopService:
    def __init__(self, store: Store, threads: dict[str, threading.Thread] | None = None) -> None:
        self.store = store
        self._job_backend = SimpleNamespace(_threads=threads or {})
        self._knowledge_graph = SimpleNamespace(
            _job_backend=SimpleNamespace(_threads={})
        )
        self._wiki_job_backend = SimpleNamespace(_threads={})
        self._memory_refresh_queue = SimpleNamespace(_threads=[])
        self._knowledge_graph_refresh_queue = SimpleNamespace(_threads=[])
        self.cancelled = threading.Event()
        self.shutdown_calls = 0

    def cancel_run(self, run_id: str) -> dict[str, str]:
        self.store.transition_run(
            run_id,
            expected_statuses={"queued", "running", "pausing"},
            status="cancelling",
        )
        self.cancelled.set()
        return {"run_id": run_id, "status": "cancelling"}

    def fail_run(self, run_id: str, message: str) -> None:
        self.store.transition_run(
            run_id,
            expected_statuses={"queued", "running", "cancelling"},
            status="error",
            error={"code": "INTERNAL_ERROR", "message": message},
        )

    def cancel_memory_dream(self, dream_id: str) -> dict[str, str]:
        row = self.store.get_memory_dream_run(dream_id) or {}
        self.store.update_memory_dream_run(
            dream_id,
            status="cancelling",
            expected_statuses={"queued", "running", "cancelling"},
            memories_created=int(row.get("memories_created") or 0),
        )
        return {"dream_run_id": dream_id, "status": "cancelling"}

    def fail_memory_dream(self, dream_id: str, message: str, *, error_code: str) -> None:
        row = self.store.get_memory_dream_run(dream_id) or {}
        current = str(row.get("status") or "")
        terminal = "cancelled" if current == "cancelling" else "error"
        self.store.update_memory_dream_run(
            dream_id,
            status=terminal,
            expected_statuses={current},
            memories_created=int(row.get("memories_created") or 0),
            error=message,
            metadata={"error_code": error_code},
        )

    def mark_knowledge_graph_rebuild_failed(self, job_id: str, *, message: str) -> None:
        self._fail_job("kg_build_jobs", job_id, message)

    def mark_knowledge_graph_wiki_refresh_failed(self, job_id: str, *, message: str) -> None:
        self._fail_job("kg_wiki_refresh_jobs", job_id, message)

    def _fail_job(self, table: str, job_id: str, message: str) -> None:
        with self.store._lock:
            self.store._conn.execute(
                f"UPDATE {table} SET status='failed',error_code='DESKTOP_PROCESS_INTERRUPTED',"
                "error_message=? WHERE id=? AND status IN ('queued','running')",
                (message, job_id),
            )
            self.store._conn.commit()

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _run(store: Store, status: str) -> str:
    run_id = store.create_run(session_id="session", mode="fast", query=status)
    store.update_run(run_id, status=status)
    return run_id


def test_active_run_cooperatively_finishes_before_store_is_closeable(tmp_path):
    store = Store(tmp_path / "desktop.sqlite3")
    run_id = _run(store, "running")
    service = FakeDesktopService(store)

    def worker() -> None:
        assert service.cancelled.wait(1.0)
        store.transition_run(
            run_id,
            expected_statuses={"cancelling"},
            status="cancelled",
        )

    thread = threading.Thread(target=worker, name="cooperative-run", daemon=True)
    service._job_backend._threads[run_id] = thread
    thread.start()

    result = shutdown_desktop_service(service, timeout_s=1.0)

    assert result.safe_to_close_store is True
    assert result.pending_threads == ()
    assert store.get_run(run_id)["status"] == "cancelled"
    assert service.shutdown_calls == 1
    store.close()


def test_blocked_worker_returns_on_deadline_and_recovery_terminalizes_after_reopen(tmp_path):
    database = tmp_path / "desktop.sqlite3"
    store = Store(database)
    run_id = _run(store, "running")
    release = threading.Event()
    thread = threading.Thread(target=lambda: release.wait(2.0), name="blocked-run", daemon=True)
    service = FakeDesktopService(store, {run_id: thread})
    thread.start()

    started = time.monotonic()
    result = shutdown_desktop_service(service, timeout_s=0.03)
    elapsed = time.monotonic() - started

    assert result.safe_to_close_store is False
    assert any("blocked-run" in name for name in result.pending_threads)
    assert elapsed < 0.5
    assert store.get_run(run_id)["status"] == "cancelling"

    release.set()
    thread.join(1.0)
    store.close()
    reopened = Store(database)
    recovery = recover_interrupted_desktop_state(FakeDesktopService(reopened))
    assert recovery.runs == 1
    assert reopened.get_run(run_id)["status"] == "error"
    reopened.close()


def test_recovery_preserves_completed_and_paused_records(tmp_path):
    store = Store(tmp_path / "desktop.sqlite3")
    completed = _run(store, "completed")
    paused = _run(store, "paused")
    interrupted = _run(store, "running")
    dream_id = store.create_memory_dream_run()
    store.upsert_post_turn_memory_run(
        run_id="post-turn",
        session_id="session",
        workspace_key="default",
        user_key="default",
        status="running",
    )
    with store._lock:
        store._conn.execute(
            "INSERT INTO kg_build_jobs(id,workspace_id,backend,status,started_at) "
            "VALUES ('graph-active','default','thread','running',1)"
        )
        store._conn.execute(
            "INSERT INTO kg_wiki_refresh_jobs(id,workspace_id,backend,status,started_at) "
            "VALUES ('wiki-active','default','thread','queued',1)"
        )
        store._conn.commit()

    result = recover_interrupted_desktop_state(FakeDesktopService(store))

    assert store.get_run(completed)["status"] == "completed"
    assert store.get_run(paused)["status"] == "paused"
    assert store.get_run(interrupted)["status"] == "error"
    assert store.get_memory_dream_run(dream_id)["status"] == "error"
    assert store.get_post_turn_memory_run("post-turn")["status"] == "failed"
    assert result.runs == 1
    assert result.memory_dreams == 1
    assert result.post_turn_memory == 1
    assert result.graph_jobs == 1
    assert result.wiki_jobs == 1
    with store._lock:
        graph = store._conn.execute(
            "SELECT status FROM kg_build_jobs WHERE id='graph-active'"
        ).fetchone()
        wiki = store._conn.execute(
            "SELECT status FROM kg_wiki_refresh_jobs WHERE id='wiki-active'"
        ).fetchone()
    assert graph["status"] == "failed"
    assert wiki["status"] == "failed"
    store.close()


def test_real_service_recovery_and_shutdown_preserve_terminal_history(monkeypatch, tmp_path):
    monkeypatch.setenv("DEEPRESEARCH_JOB_BACKEND", "thread")
    monkeypatch.setenv("DEEPRESEARCH_VECTOR_BACKEND", "numpy")
    service = BaselineService(
        provider=ProviderSettings(),
        db_path=tmp_path / "desktop.sqlite3",
        snapshot_root=tmp_path / "snapshots",
    )
    store = service.store
    queued = _run(store, "queued")
    running = _run(store, "running")
    pausing = _run(store, "pausing")
    completed = _run(store, "completed")
    paused = _run(store, "paused")
    with store._lock:
        store._conn.execute(
            "INSERT INTO kg_build_jobs(id,workspace_id,backend,status,started_at) "
            "VALUES ('real-graph','default','thread','running',1)"
        )
        store._conn.execute(
            "INSERT INTO kg_wiki_refresh_jobs(id,workspace_id,backend,status,started_at) "
            "VALUES ('real-wiki','default','thread','queued',1)"
        )
        store._conn.commit()

    recovery = recover_interrupted_desktop_state(service)

    assert recovery.runs == 3
    for run_id in (queued, running, pausing):
        row = store.get_run(run_id)
        assert row["status"] == "error"
        assert row["error"]["error"]["code"] == "INTERNAL_ERROR"
    assert store.get_run(completed)["status"] == "completed"
    assert store.get_run(paused)["status"] == "paused"
    assert service.get_knowledge_graph_rebuild_job("real-graph")["status"] == "failed"
    assert service.get_knowledge_graph_wiki_refresh_job("real-wiki")["status"] == "failed"

    shutdown_orphan = _run(store, "queued")
    shutdown = shutdown_desktop_service(service, timeout_s=1.0)
    assert shutdown.safe_to_close_store is True
    assert shutdown.pending_threads == ()
    assert store.get_run(shutdown_orphan)["status"] == "error"
    assert store.get_run(completed)["status"] == "completed"
    assert store.get_run(paused)["status"] == "paused"
    store.close()
