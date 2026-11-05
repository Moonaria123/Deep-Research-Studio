"""T15 post-run memory refresh trigger contract."""

from __future__ import annotations

from deepresearch_core.baseline import BaselineService
from deepresearch_core.baseline.budget import Mode, RunRequest
from deepresearch_core.baseline.citations import GroundedAnswer
from deepresearch_core.baseline.orchestrator import RunOutcome
from deepresearch_core.settings import ProviderSettings
from tests.baseline.fakes import FakeLLM


def _service(memory_llm: FakeLLM) -> BaselineService:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    service.configure_runtime_for_tests(llm=FakeLLM(), memory_llm=memory_llm, memory_refresh_inline=True)
    return service


def _request(session_id: str) -> RunRequest:
    return RunRequest(
        query="Post-run memory refresh",
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


def _persist_turn_messages(service: BaselineService, session_id: str, run_id: str) -> None:
    service.store.add_message(session_id, "user", "Research request", metadata={"run_id": run_id})
    service.store.add_message(session_id, "assistant", "Delivered answer", metadata={"run_id": run_id})


def test_completed_run_triggers_memory_refresh():
    service = _service(
        FakeLLM(
            scripted={
                "Capture the last completed DeepResearch turn": '{"summary":"Post-run summary persisted."}',
                "Distill a user memory profile": '{"sections":[]}',
            }
        )
    )
    session_id = service.store.create_session(title="Refresh")
    run_id = service.store.create_run(session_id=session_id, mode="fast", query="Refresh")
    _persist_turn_messages(service, session_id, run_id)

    service._persist_library_artifacts(_request(session_id), _outcome(run_id))

    rows = service.store.list_memories(namespace="workspace", scope_key="default", limit=20)
    row = [row for row in rows if row["metadata"].get("session_id") == session_id][0]
    assert row["metadata"]["source"] == "post_turn_memory_v1"
    assert row["content"] == "Post-run summary persisted."


def test_degraded_and_partial_timeout_with_result_trigger_but_terminal_failures_do_not():
    service = _service(
        FakeLLM(
            scripted={
                "Capture the last completed DeepResearch turn": '{"summary":"Partial answer summary."}',
                "Distill a user memory profile": '{"sections":[]}',
            }
        )
    )
    session_id = service.store.create_session(title="Partial")
    partial_id = service.store.create_run(session_id=session_id, mode="fast", query="Partial")
    degraded_id = service.store.create_run(session_id=session_id, mode="fast", query="Degraded")
    error_id = service.store.create_run(session_id=session_id, mode="fast", query="Error")
    cancelled_id = service.store.create_run(session_id=session_id, mode="fast", query="Cancelled")
    no_result_id = service.store.create_run(session_id=session_id, mode="fast", query="No result")

    for run_id in (partial_id, degraded_id, error_id, cancelled_id, no_result_id):
        _persist_turn_messages(service, session_id, run_id)

    request = _request(session_id)
    service._persist_library_artifacts(request, _outcome(partial_id, status="partial_timeout", text="Partial answer"))
    service._persist_library_artifacts(
        request,
        _outcome(degraded_id, status="completed_degraded", text="Bounded answer"),
    )
    service._persist_library_artifacts(request, _outcome(error_id, status="error", text="Error answer"))
    service._persist_library_artifacts(request, _outcome(cancelled_id, status="cancelled", text="Cancelled answer"))
    service._persist_library_artifacts(request, _outcome(no_result_id, status="partial_timeout", text=""))

    rows = [
        row
        for row in service.store.list_memories(namespace="workspace", scope_key="default", limit=20)
        if row["metadata"].get("source") == "post_turn_memory_v1"
    ]
    assert len(rows) == 2
    assert {row["metadata"]["run_id"] for row in rows} == {partial_id, degraded_id}
