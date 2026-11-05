"""T16 post-run Knowledge Graph refresh trigger contract."""

from __future__ import annotations

import time

from deepresearch_core.baseline import BaselineService
from deepresearch_core.baseline.budget import Mode, RunRequest
from deepresearch_core.baseline.citations import GroundedAnswer
from deepresearch_core.baseline.knowledge_graph.store_ops import KnowledgeGraphStore
from deepresearch_core.baseline.orchestrator import RunOutcome
from deepresearch_core.settings import ProviderSettings
from tests.baseline.fakes import FakeLLM


def _service() -> BaselineService:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    service.configure_runtime_for_tests(
        llm=FakeLLM(),
        memory_llm=FakeLLM(
            scripted={
                "Distill a session research summary": '{"summary":"KG refresh source summary."}',
                "Distill a user memory profile": '{"sections":[]}',
            }
        ),
        memory_refresh_inline=True,
        knowledge_graph_refresh_inline=True,
    )
    return service


def _request(session_id: str) -> RunRequest:
    return RunRequest(
        query="Post-run KG refresh",
        mode=Mode.FAST,
        session_id=session_id,
        workspace_key="default",
        user_key="default",
    )


def _outcome(run_id: str, *, status: str = "completed", text: str = "Useful terminal answer") -> RunOutcome:
    return RunOutcome(
        run_id=run_id,
        mode=Mode.FAST.value,
        status=status,
        answer=GroundedAnswer(text=text, citations=[], claim_map={}, coverage=0.0),
    )


def _wait_for_job(service: BaselineService, expected_status: str = "succeeded") -> dict:
    for _ in range(50):
        job = service.get_knowledge_graph_status()["last_build_job"]
        if job and job["status"] == expected_status:
            return job
        time.sleep(0.02)
    raise AssertionError(f"knowledge graph job did not reach {expected_status}")


def test_completed_degraded_and_partial_timeout_runs_trigger_knowledge_graph_refresh() -> None:
    service = _service()
    session_id = service.store.create_session(title="KG refresh")
    completed_id = service.store.create_run(session_id=session_id, mode="fast", query="Completed")
    partial_id = service.store.create_run(session_id=session_id, mode="fast", query="Partial")
    degraded_id = service.store.create_run(session_id=session_id, mode="fast", query="Degraded")

    service._persist_library_artifacts(_request(session_id), _outcome(completed_id))
    first_job = _wait_for_job(service)
    service._persist_library_artifacts(
        _request(session_id),
        _outcome(partial_id, status="partial_timeout", text="Partial answer"),
    )
    second_job = _wait_for_job(service)
    service._persist_library_artifacts(
        _request(session_id),
        _outcome(degraded_id, status="completed_degraded", text="Bounded answer"),
    )
    third_job = _wait_for_job(service)

    assert first_job["id"] != second_job["id"]
    assert second_job["id"] != third_job["id"]
    graph = service.get_knowledge_graph()
    assert any(node["type"] == "session" for node in graph["nodes"])
    service.store.close()


def test_knowledge_graph_refresh_respects_setting_and_running_conflict() -> None:
    service = _service()
    session_id = service.store.create_session(title="KG disabled")
    disabled_run_id = service.store.create_run(session_id=session_id, mode="fast", query="Disabled")
    service.update_knowledge_graph_config({"post_run_rebuild_enabled": False})

    service._persist_library_artifacts(_request(session_id), _outcome(disabled_run_id))
    assert service.get_knowledge_graph_status()["last_build_job"] is None

    service.update_knowledge_graph_config({"post_run_rebuild_enabled": True})
    ops = KnowledgeGraphStore(service.store)
    version_id = ops.create_staging_version(workspace_id="default")
    running = ops.create_build_job(workspace_id="default", backend="thread")
    ops.mark_build_job_running(running["id"], version_id=version_id)
    conflict_run_id = service.store.create_run(session_id=session_id, mode="fast", query="Conflict")

    service._persist_library_artifacts(_request(session_id), _outcome(conflict_run_id))

    rows = service.store._conn.execute(  # noqa: SLF001 - precise job-count contract
        "SELECT id FROM kg_build_jobs WHERE workspace_id='default'"
    ).fetchall()
    assert [row["id"] for row in rows] == [running["id"]]
    service.store.close()
