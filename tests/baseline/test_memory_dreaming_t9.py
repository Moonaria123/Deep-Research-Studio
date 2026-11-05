"""T15: Dreaming wrapper delegates to memory continuity distillation."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.settings import ProviderSettings
from tests.baseline.fakes import FakeLLM

pytest.importorskip("fastapi.testclient", reason="fastapi not installed")


def _service_with_memory_llm(llm: FakeLLM, *, config_path=None) -> BaselineService:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", config_path=config_path)
    service.configure_runtime_for_tests(llm=FakeLLM(), memory_llm=llm)
    return service


def test_dreaming_distills_session_summary_and_user_profile():
    llm = FakeLLM(
        scripted={
            "Distill a session research summary": '{"summary":"Session summary: local-first citation governance."}',
            "Distill a user memory profile": (
                '{"sections":[{"id":"preferences","title":"Research Preferences",'
                '"body":"Prefers concise evidence-backed answers."}]}'
            ),
        }
    )
    service = _service_with_memory_llm(llm)
    sid = service.store.create_session(title="Local research")
    service.store.add_message(sid, "user", "Compare local-first research workflows")
    service.store.add_message(sid, "assistant", "User prefers concise evidence-backed answers.")

    result = service.start_memory_dream(
        {"workspace_key": "default", "user_key": "default", "session_id": sid},
        background=False,
    )

    assert result["status"] == "completed"
    assert result["memories_created"] == 2
    workspace = service.store.list_memories(namespace="workspace", scope_key="default", limit=10)
    assert [m for m in workspace if m["metadata"].get("source") == "session_distill_v1"][0]["metadata"][
        "session_id"
    ] == sid
    user_profile = service.store.list_memories(namespace="user", scope_key="default", limit=10)[0]
    assert user_profile["metadata"]["source"] == "user_profile_v1"
    assert user_profile["metadata"]["sections"][0]["title"] == "Research Preferences"


def test_dreaming_prompt_excludes_archived_memories():
    llm = FakeLLM(
        scripted={
            "Distill a session research summary": '{"summary":"active-blue-needle"}',
            "Distill a user memory profile": '{"sections":[]}',
        }
    )
    service = _service_with_memory_llm(llm)
    service.add_memory({"layer": "workspace", "scope_key": "default", "content": "active-blue-needle"})
    archived_id = service.add_memory(
        {"layer": "workspace", "scope_key": "default", "content": "archived-red-needle"}
    )["memory_id"]
    service.archive_memory(archived_id)
    sid = service.store.create_session(title="Needles")
    service.store.add_message(sid, "user", "Use active memory")

    result = service.start_memory_dream({"workspace_key": "default", "session_id": sid}, background=False)

    assert result["status"] == "completed"
    prompt = llm.calls[0][-1]["content"]
    assert "active-blue-needle" in prompt
    assert "archived-red-needle" not in prompt


def test_dreaming_failure_marks_status_without_creating_memory():
    llm = FakeLLM(scripted={"Distill a session research summary": "not json"})
    service = _service_with_memory_llm(llm)

    result = service.start_memory_dream({"workspace_key": "default"}, background=False)

    assert result["status"] == "error"
    assert result["memories_created"] == 0
    assert result["error"]
    assert service.recall_memory({"workspace_key": "default", "user_key": "", "limit": 10}) == []


def test_dreaming_requires_memory_processing_role():
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    service.configure_runtime_for_tests(llm=FakeLLM())

    with pytest.raises(BaselineError) as exc:
        service.start_memory_dream({"workspace_key": "default"}, background=False)

    assert exc.value.code is ErrorCode.PROVIDER_NOT_CONFIGURED


def test_memory_dream_api_status_and_trigger():
    from fastapi.testclient import TestClient

    llm = FakeLLM(
        scripted={
            "Distill a session research summary": '{"summary":"library RRF"}',
            "Distill a user memory profile": '{"sections":[]}',
        }
    )
    service = _service_with_memory_llm(llm)
    sid = service.store.create_session(title="Library RRF")
    service.store.add_message(sid, "user", "Remember library RRF")
    client = TestClient(create_app(service))

    empty = client.get("/api/v1/memory/dream").json()
    assert empty["status"] == "idle"

    triggered = client.post(
        "/api/v1/memory/dream",
        json={"workspace_key": "default", "session_id": sid, "background": False},
    ).json()

    assert triggered["status"] == "completed"
    assert triggered["memories_created"] == 1
    assert triggered["structured_strategy"] == "json_schema_strict"
    assert triggered["incomplete"] is False
    status = client.get("/api/v1/memory/dream").json()
    assert status["dream_run_id"] == triggered["dream_run_id"]
    assert status["provider_ref"] == ""
    assert "storage_path" not in status


def test_memory_dream_schedule_saves_and_does_not_trigger_before_due(tmp_path):
    llm = FakeLLM(
        scripted={
            "Distill a session research summary": '{"summary":"library RRF"}',
            "Distill a user memory profile": '{"sections":[]}',
        }
    )
    service = _service_with_memory_llm(llm, config_path=tmp_path / "config.json")
    sid = service.store.create_session(title="Library RRF")
    service.store.add_message(sid, "user", "Remember library RRF")

    completed = service.start_memory_dream({"workspace_key": "default", "session_id": sid}, background=False)
    schedule = service.update_memory_dream_schedule({"schedule": "daily"})
    status = service.get_memory_dream_status({"workspace_key": "default", "user_key": "default"})

    assert schedule["schedule"] == "daily"
    assert status["schedule"] == "daily"
    assert status["dream_run_id"] == completed["dream_run_id"]
    assert status["next_scheduled_at"] > status["finished_at"]


def test_memory_dream_schedule_api_updates_status(tmp_path):
    from fastapi.testclient import TestClient

    service = _service_with_memory_llm(FakeLLM(), config_path=tmp_path / "config.json")
    client = TestClient(create_app(service))

    updated = client.put("/api/v1/memory/dream/schedule", json={"schedule": "weekly"}).json()
    status = client.get("/api/v1/memory/dream").json()

    assert updated["schedule"] == "weekly"
    assert status["schedule"] == "weekly"
