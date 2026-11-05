"""T15 workspace session memory distillation contract."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline import BaselineService
from deepresearch_core.baseline.adapters import LLMResult
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.settings import ProviderSettings
from tests.baseline.fakes import FakeLLM


def _service(memory_llm: FakeLLM | None = None) -> BaselineService:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    kwargs = {"llm": FakeLLM()}
    if memory_llm is not None:
        kwargs["memory_llm"] = memory_llm
    service.configure_runtime_for_tests(**kwargs)
    return service


def _memory_llm(summary: str = "Session summary: citation-first local research.") -> FakeLLM:
    return FakeLLM(
        scripted={
            "Distill a session research summary": f'{{"summary":"{summary}"}}',
            "Distill a user memory profile": '{"sections":[]}',
        }
    )


def _enable_memory_thinking(service: BaselineService, *, budget: int = 2048) -> None:
    service.upsert_llm_provider(
        {
            "id": "memory-provider",
            "type": "openai",
            "base_url": "https://memory.example/v1",
            "models": [{"id": "writer", "kind": "chat", "thinking_budget": budget}],
        }
    )
    service.set_llm_roles({"memory": "memory-provider:writer"})
    service.update_memory_settings({"memory_thinking_enabled": True})


class _MemoryDistillLLM(FakeLLM):
    def __init__(self, replies: dict[str, str]) -> None:
        super().__init__(scripted={})
        self.replies = replies
        self.kwargs: list[dict] = []

    def complete(self, messages, **kwargs):  # type: ignore[override]
        self.calls.append(list(messages))
        self.kwargs.append(kwargs)
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        needles = sorted(self.replies, key=lambda item: 0 if item.startswith("Distill a user memory profile") else 1)
        for needle in needles:
            reply = self.replies[needle]
            if needle in user:
                return LLMResult(text=reply, model=self.model)
        return LLMResult(text='{"sections":[]}', model=self.model)


def test_workspace_distill_keeps_one_current_memory_per_session():
    service = _service(_memory_llm("Session summary: local-first workflows."))
    session_id = service.store.create_session(title="Local workflows")
    service.store.add_message(session_id, "user", "Compare local-first workflows")
    service.store.add_message(session_id, "assistant", "The user cares about local storage.")

    first = service.start_memory_dream({"workspace_key": "default", "session_id": session_id}, background=False)
    second = service.start_memory_dream({"workspace_key": "default", "session_id": session_id}, background=False)

    rows = [
        row
        for row in service.store.list_memories(namespace="workspace", scope_key="default", limit=20)
        if row["metadata"].get("source") == "session_distill_v1"
    ]
    assert first["status"] == "completed"
    assert second["status"] == "completed"
    assert len(rows) == 1
    assert rows[0]["metadata"]["session_id"] == session_id
    assert "local-first workflows" in rows[0]["content"]


def test_workspace_distill_creates_one_memory_per_session():
    service = _service(_memory_llm("Session summary: repeated summary template."))
    first_session = service.store.create_session(title="First")
    second_session = service.store.create_session(title="Second")

    service.start_memory_dream({"workspace_key": "default", "session_id": first_session}, background=False)
    service.start_memory_dream({"workspace_key": "default", "session_id": second_session}, background=False)

    rows = [
        row
        for row in service.store.list_memories(namespace="workspace", scope_key="default", limit=20)
        if row["metadata"].get("source") == "session_distill_v1"
    ]
    assert {row["metadata"]["session_id"] for row in rows} == {first_session, second_session}


def test_workspace_distill_parses_fenced_json_and_requests_structured_output():
    memory_llm = _MemoryDistillLLM(
        {
            "Session: JSON fence": 'Here is the JSON:\n```json\n{"summary":"Session summary: fenced JSON works.",}\n```\nDone.',
            "Distill a user memory profile": '{"sections":[]}',
        }
    )
    service = _service(memory_llm)
    session_id = service.store.create_session(title="JSON fence")

    result = service.start_memory_dream({"workspace_key": "default", "session_id": session_id}, background=False)

    assert result["status"] == "completed"
    assert result["error"] is None
    assert memory_llm.kwargs[0]["thinking_budget"] == 0
    assert memory_llm.kwargs[0]["response_format"]["type"] == "json_schema"
    rows = service.store.list_memories(namespace="workspace", scope_key="default", limit=10)
    assert any("fenced JSON works" in row["content"] for row in rows)


def test_workspace_distill_repairs_schema_failure_once():
    memory_llm = _MemoryDistillLLM(
        {
            "Session: Repair": '{"wrong":"shape"}',
            "Error class: SCHEMA_VALIDATION_FAILED": '{"summary":"Session summary: repaired schema succeeds."}',
            "Distill a user memory profile": '{"sections":[]}',
        }
    )
    service = _service(memory_llm)
    session_id = service.store.create_session(title="Repair")

    result = service.start_memory_dream({"workspace_key": "default", "session_id": session_id}, background=False)

    assert result["status"] == "completed"
    assert len(memory_llm.kwargs) == 3
    rows = service.store.list_memories(namespace="workspace", scope_key="default", limit=10)
    assert any("repaired schema succeeds" in row["content"] for row in rows)


def test_workspace_distill_uses_memory_profile_thinking_when_enabled():
    memory_llm = _MemoryDistillLLM(
        {
            "Distill a session research summary": '{"summary":"Session summary: thinking path."}',
            "Distill a user memory profile": '{"sections":[]}',
        }
    )
    service = _service(memory_llm)
    _enable_memory_thinking(service, budget=4096)
    session_id = service.store.create_session(title="Thinking")

    result = service.start_memory_dream({"workspace_key": "default", "session_id": session_id}, background=False)

    assert result["status"] == "completed"
    assert memory_llm.kwargs
    assert all(call["thinking_budget"] == 4096 for call in memory_llm.kwargs)


def test_workspace_distill_incremental_skips_current_session_without_llm():
    service = _service(_memory_llm("Session summary: stable current summary."))
    session_id = service.store.create_session(title="Incremental")
    service.store.add_message(session_id, "user", "Keep this summary current.")

    first = service.start_memory_dream({"workspace_key": "default", "session_id": session_id}, background=False)
    memory_llm = _MemoryDistillLLM({"Distill a session research summary": '{"summary":"should not run"}'})
    service.configure_runtime_for_tests(llm=FakeLLM(), memory_llm=memory_llm)
    second = service.start_memory_dream({"workspace_key": "default", "session_id": session_id}, background=False)

    assert first["status"] == "completed"
    assert second["status"] == "completed"
    assert second["workspace_skipped"] == 1
    assert second["memories_created"] == 0
    assert len(memory_llm.kwargs) == 1
    assert "Distill a user memory profile" in memory_llm.calls[0][-1]["content"]
    assert "Distill a session research summary" not in memory_llm.calls[0][-1]["content"]


def test_workspace_distill_skips_bad_session_json_and_preserves_existing_block():
    memory_llm = _MemoryDistillLLM(
        {
            "Session: Bad JSON": "{not valid json",
            "Session: Good JSON": '{"summary":"Session summary: good JSON survives."}',
            "Distill a user memory profile": (
                '{"sections":[{"id":"preferences","title":"Research Preferences","body":"Prefers stable JSON models."}]}'
            ),
        }
    )
    service = _service(memory_llm)
    bad_session = service.store.create_session(title="Bad JSON")
    good_session = service.store.create_session(title="Good JSON")
    service.store.upsert_workspace_session_memory(
        session_id=bad_session,
        workspace_key="default",
        content="Session summary: old safe block remains.",
        metadata={"session_title": "Bad JSON"},
    )

    result = service.start_memory_dream({"workspace_key": "default"}, background=False)

    assert result["status"] == "completed"
    assert result["error_code"] == "PARTIAL_FAILURE"
    assert result["failed_stage"] == "session_distill"
    assert result["failed_session_id"] == bad_session
    assert result["failed_sessions"] == 1
    rows = service.store.list_memories(namespace="workspace", scope_key="default", limit=20)
    contents = [row["content"] for row in rows]
    assert any("old safe block remains" in content for content in contents)
    assert any("good JSON survives" in content for content in contents)
    assert not any("not valid json" in content for content in contents)
    assert service.store.list_memories(namespace="user", scope_key="default", limit=5)


def test_workspace_distill_all_failures_exposes_stage_error():
    memory_llm = _MemoryDistillLLM(
        {
            "Distill a session research summary": "{not valid json",
            "Distill a user memory profile": "{also not valid",
            "Error class: JSON_PARSE_FAILED": "{still invalid",
        }
    )
    service = _service(memory_llm)
    session_id = service.store.create_session(title="Broken")

    result = service.start_memory_dream({"workspace_key": "default", "session_id": session_id}, background=False)

    assert result["status"] == "error"
    assert result["error_code"] == "MEMORY_DISTILLATION_ALL_FAILED"
    assert result["failed_stage"] == "session_distill"
    assert result["failed_session_id"] == session_id
    assert result["attempted_sessions"] == 1
    assert result["failed_sessions"] == 1
    assert result["user_profile_failed"] is True


def test_workspace_distill_requires_memory_role():
    service = _service()

    with pytest.raises(BaselineError) as exc:
        service.start_memory_dream({"workspace_key": "default"}, background=False)

    assert exc.value.code is ErrorCode.PROVIDER_NOT_CONFIGURED


def test_post_turn_memory_refresh_writes_turn_artifact_only():
    memory_llm = _MemoryDistillLLM(
        {
            "Capture the last completed DeepResearch turn": '{"summary":"Session summary: current turn only."}',
            "Distill a session research summary": '{"summary":"should not full distill"}',
            "Distill a user memory profile": '{"sections":[]}',
        }
    )
    service = _service(memory_llm)
    session_id = service.store.create_session(title="Turn capture")
    service.store.add_message(session_id, "user", "Remember only this turn.", metadata={"run_id": "run-turn-1"})
    service.store.add_message(session_id, "assistant", "Turn output suitable for memory.", metadata={"run_id": "run-turn-1"})

    result = service.refresh_post_turn_memory(
        session_id=session_id,
        run_id="run-turn-1",
        workspace_key="default",
        user_key="default",
    )

    assert result["status"] == "completed"
    assert result["source"] == "post_turn_memory_v1"
    rows = service.store.list_memories(namespace="workspace", scope_key="default", limit=20)
    turn_rows = [row for row in rows if row["metadata"].get("source") == "post_turn_memory_v1"]
    session_rows = [row for row in rows if row["metadata"].get("source") == "session_distill_v1"]
    assert len(turn_rows) == 1
    assert turn_rows[0]["metadata"]["session_id"] == session_id
    assert turn_rows[0]["metadata"]["run_id"] == "run-turn-1"
    assert "current turn only" in turn_rows[0]["content"]
    assert session_rows == []
    assert all("Distill a user memory profile" not in call[-1]["content"] for call in memory_llm.calls)


def test_post_turn_memory_refresh_uses_memory_profile_thinking_when_enabled():
    memory_llm = _MemoryDistillLLM(
        {
            "Capture the last completed DeepResearch turn": '{"summary":"Session summary: thinking turn."}',
        }
    )
    service = _service(memory_llm)
    _enable_memory_thinking(service, budget=1024)
    session_id = service.store.create_session(title="Turn thinking")
    service.store.add_message(session_id, "user", "Remember this turn.", metadata={"run_id": "run-thinking"})
    service.store.add_message(session_id, "assistant", "Turn output suitable for memory.", metadata={"run_id": "run-thinking"})

    result = service.refresh_post_turn_memory(session_id=session_id, run_id="run-thinking")

    assert result["status"] == "completed"
    assert memory_llm.kwargs
    assert all(call["thinking_budget"] == 1024 for call in memory_llm.kwargs)


def test_delete_session_cascades_post_turn_artifact_without_touching_other_sources():
    service = _service(_memory_llm())
    session_id = service.store.create_session(title="Delete cascade")
    keep_session = service.store.create_session(title="Keep")
    service.store.add_memory(
        namespace="workspace",
        scope_key="default",
        content="Turn artifact to delete",
        metadata={"source": "post_turn_memory_v1", "session_id": session_id, "run_id": "run-delete"},
    )
    service.store.upsert_workspace_session_memory(
        session_id=session_id,
        workspace_key="default",
        content="Session summary to delete",
        metadata={},
    )
    service.store.add_memory(
        namespace="workspace",
        scope_key="default",
        content="Other turn artifact remains",
        metadata={"source": "post_turn_memory_v1", "session_id": keep_session, "run_id": "run-keep"},
    )

    service.delete_session(session_id)

    rows = service.store.list_memories(namespace="workspace", scope_key="default", limit=20)
    assert all(row["metadata"].get("session_id") != session_id for row in rows)
    assert any(row["metadata"].get("session_id") == keep_session for row in rows)


def test_memory_write_audit_reports_latest_safe_status():
    service = _service(_memory_llm("Session summary: audit-ready memory."))
    session_id = service.store.create_session(title="Audit")
    service.store.add_message(session_id, "user", "Audit this memory update.")

    dream = service.start_memory_dream({"workspace_key": "default", "session_id": session_id}, background=False)
    audit = service.get_memory_write_audit({"workspace_key": "default", "user_key": "default"})

    assert dream["status"] == "completed"
    assert audit["source"] == "memory_continuity_v1"
    assert audit["attempted"] == 1
    assert audit["succeeded"] == 1
    assert audit["failed"] == 0
    assert audit["checked_at"]
    assert "raw" not in str(audit).lower()
    assert "sk-" not in str(audit)


def test_memory_fact_v1_add_only_deduplicates_same_session_content():
    memory_llm = _MemoryDistillLLM(
        {
            "Distill a session research summary": '{"summary":"Session summary: fact path."}',
            "Distill a user memory profile": '{"sections":[]}',
            "Extract ADD-only memory_fact_v1 candidates": (
                '{"memories":[{"content":"Prefers local-first memory.","kind":"preference","confidence":"high"}]}'
            ),
        }
    )
    service = _service(memory_llm)
    session_id = service.store.create_session(title="Fact v1")
    service.store.add_message(session_id, "user", "Remember that I prefer local-first memory.")

    first = service.start_memory_dream(
        {"workspace_key": "default", "session_id": session_id, "include_facts": True},
        background=False,
    )
    second = service.start_memory_dream(
        {"workspace_key": "default", "session_id": session_id, "include_facts": True},
        background=False,
    )
    audit = service.get_memory_write_audit({"workspace_key": "default"})

    rows = [
        row
        for row in service.store.list_memories(namespace="workspace", scope_key="default", limit=20)
        if row["metadata"].get("source") == "memory_fact_v1"
    ]
    assert first["accepted"] == 1
    assert second["accepted"] == 0
    assert second["skipped"] >= 1
    assert len(rows) == 1
    assert rows[0]["metadata"]["schema_version"] == "memory_fact_v1"
    assert audit["source"] == "memory_fact_v1"
    assert audit["attempted"] == 1
    assert audit["succeeded"] == 0
    assert audit["skipped"] >= 1


def test_memory_fact_v1_uses_memory_profile_thinking_when_enabled():
    memory_llm = _MemoryDistillLLM(
        {
            "Extract ADD-only memory_fact_v1 candidates": (
                '{"memories":[{"content":"Prefers explicit memory thinking.","kind":"preference","confidence":"high"}]}'
            ),
        }
    )
    service = _service(memory_llm)
    _enable_memory_thinking(service, budget=8192)
    session_id = service.store.create_session(title="Fact thinking")
    service.store.add_message(session_id, "user", "Remember that explicit memory thinking matters.")

    result = service._memory_distiller.refresh_memory_facts(
        workspace_key="default",
        user_key="default",
        session_id=session_id,
        run_id="run-fact-thinking",
    )

    assert result["accepted"] == 1
    assert result["source"] == "memory_fact_v1"
    assert result["failed"] == 0
    assert memory_llm.kwargs
    assert all(call["thinking_budget"] == 8192 for call in memory_llm.kwargs)
