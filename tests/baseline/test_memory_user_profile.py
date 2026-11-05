"""T15 user profile memory distillation contract."""

from __future__ import annotations

from deepresearch_core.baseline import BaselineService
from deepresearch_core.baseline.budget import Mode, RunRequest
from deepresearch_core.baseline.context_budget import default_config
from deepresearch_core.settings import ProviderSettings
from tests.baseline.fakes import FakeLLM


def _service(memory_llm: FakeLLM) -> BaselineService:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    service.configure_runtime_for_tests(llm=FakeLLM(), memory_llm=memory_llm)
    return service


def _profile_llm(body: str = "Prefers concise evidence-backed answers.") -> FakeLLM:
    return FakeLLM(
        scripted={
            "Distill a session research summary": '{"summary":"Session summary: evidence workflows."}',
            "Distill a user memory profile": (
                '{"sections":['
                '{"id":"preferences","title":"Research Preferences","body":"'
                + body
                + '"},'
                '{"id":"constraints","title":"Constraints","body":"Avoid non-local storage."}'
                "]}"
            ),
        }
    )


def test_user_profile_distill_keeps_one_current_profile_with_sections():
    service = _service(_profile_llm())

    service.start_memory_dream({"workspace_key": "default", "user_key": "default"}, background=False)
    service.start_memory_dream({"workspace_key": "default", "user_key": "default"}, background=False)

    rows = [
        row
        for row in service.store.list_memories(namespace="user", scope_key="default", limit=20)
        if row["metadata"].get("source") == "user_profile_v1"
    ]
    assert len(rows) == 1
    assert rows[0]["metadata"]["sections"][0]["title"] == "Research Preferences"
    assert "## Research Preferences" in rows[0]["content"]
    assert "Avoid non-local storage" in rows[0]["content"]


def test_memory_context_honors_workspace_vs_workspace_user_policy():
    service = _service(_profile_llm("Prefers matrix-style summaries."))
    service.start_memory_dream({"workspace_key": "default", "user_key": "default"}, background=False)
    cfg = default_config()

    with_user = service._orchestrator._memory_context(
        RunRequest(
            query="research preferences",
            mode=Mode.FAST,
            workspace_key="default",
            user_key="default",
            memory_policy="workspace_user",
        ),
        cfg,
        budget=1200,
    )
    workspace_only = service._orchestrator._memory_context(
        RunRequest(
            query="research preferences",
            mode=Mode.FAST,
            workspace_key="default",
            user_key="default",
            memory_policy="workspace",
        ),
        cfg,
        budget=1200,
    )

    assert "Prefers matrix-style summaries" in with_user
    assert "Prefers matrix-style summaries" not in workspace_only
