"""T18 Memory settings API and config tests."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.baseline.budget import RunRequest
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.settings import ProviderSettings
from tests.baseline.fakes import FakeLLM

pytest.importorskip("fastapi.testclient", reason="fastapi not installed")


def _service(*, config_path=None, memory_llm: FakeLLM | None = None) -> BaselineService:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", config_path=config_path)
    service.configure_runtime_for_tests(llm=FakeLLM(), memory_llm=memory_llm)
    return service


def test_memory_settings_defaults():
    service = _service()

    settings = service.get_memory_settings()

    assert settings["post_turn_memory_refresh_enabled"] is True
    assert settings["workspace_memory_refresh_strategy"] == "incremental"
    assert settings["memory_thinking_enabled"] is False
    assert settings["format_probe"] is None


def test_memory_settings_put_roundtrip_preserves_other_config_blocks(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"providers": {"existing": True}}), encoding="utf-8")
    service = _service(config_path=config_path)

    updated = service.update_memory_settings(
        {
            "post_turn_memory_refresh_enabled": False,
            "workspace_memory_refresh_strategy": "full",
            "memory_thinking_enabled": True,
        }
    )
    reloaded = _service(config_path=config_path).get_memory_settings()
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    assert updated["post_turn_memory_refresh_enabled"] is False
    assert updated["memory_thinking_enabled"] is True
    assert reloaded["workspace_memory_refresh_strategy"] == "full"
    assert reloaded["memory_thinking_enabled"] is True
    assert raw["providers"] == {"existing": True}


def test_memory_settings_rejects_invalid_strategy():
    service = _service()

    with pytest.raises(BaselineError) as exc:
        service.update_memory_settings({"workspace_memory_refresh_strategy": "sometimes"})

    assert exc.value.code is ErrorCode.INVALID_REQUEST


def test_memory_settings_rejects_invalid_thinking_toggle():
    service = _service()

    with pytest.raises(BaselineError) as exc:
        service.update_memory_settings({"memory_thinking_enabled": "yes"})

    assert exc.value.code is ErrorCode.INVALID_REQUEST


def test_memory_settings_api_and_probe_cache(tmp_path):
    from fastapi.testclient import TestClient

    llm = FakeLLM(scripted={"Return only JSON for this DeepResearch memory format test": '{"summary":"Probe ok."}'})
    service = _service(config_path=tmp_path / "config.json", memory_llm=llm)
    client = TestClient(create_app(service))

    defaults = client.get("/api/v1/settings/memory").json()
    updated = client.put("/api/v1/settings/memory", json={"post_turn_memory_refresh_enabled": False}).json()
    probe = client.post("/api/v1/settings/memory/format-probe").json()
    settings = client.get("/api/v1/settings/memory").json()

    assert defaults["workspace_memory_refresh_strategy"] == "incremental"
    assert defaults["memory_thinking_enabled"] is False
    assert updated["post_turn_memory_refresh_enabled"] is False
    assert probe["status"] == "passed"
    assert settings["format_probe"]["status"] == "passed"


def test_memory_write_audit_api_is_secret_free(tmp_path):
    from fastapi.testclient import TestClient

    service = _service(config_path=tmp_path / "config.json", memory_llm=_ProbeLLMForAudit())
    session_id = service.store.create_session(title="Audit API")
    service.store.add_message(session_id, "user", "Keep audit API safe.")
    client = TestClient(create_app(service))

    dream = client.post(
        "/api/v1/memory/dream",
        json={"workspace_key": "default", "user_key": "default", "session_id": session_id, "background": False},
    ).json()
    audit = client.get("/api/v1/memory/write-audit").json()

    assert dream["status"] == "completed"
    assert audit["source"] == "memory_continuity_v1"
    assert audit["attempted"] == 1
    assert audit["succeeded"] == 1
    assert audit["failed"] == 0
    assert "sk-" not in json.dumps(audit)


class _ProbeLLMForAudit(FakeLLM):
    def complete(self, messages, **kwargs):  # type: ignore[override]
        self.calls.append(list(messages))
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if "Distill a session research summary" in user:
            return SimpleNamespace(text='{"summary":"Session summary: audit API safe."}', finish_reason="stop", model=self.model)
        if "Distill a user memory profile" in user:
            return SimpleNamespace(text='{"sections":[]}', finish_reason="stop", model=self.model)
        return SimpleNamespace(text='{"summary":"ok"}', finish_reason="stop", model=self.model)


def test_memory_probe_cache_marks_stale_when_memory_role_changes(tmp_path):
    service = _service(config_path=tmp_path / "config.json")
    service.upsert_llm_provider(
        {"id": "p1", "type": "openai", "base_url": "https://one.example/v1", "models": ["m1", "m2"]}
    )
    service.set_llm_roles({"memory": "p1:m1"})
    service.update_memory_settings(
        {
            "format_probe": {
                "status": "passed",
                "provider_ref": "p1:m1",
                "capability": "json_schema_strict",
                "structured_strategy": "json_schema_strict",
                "checked_at": 1,
                "diagnostics": {"response_shape": "json_object"},
            }
        }
    )

    service.set_llm_roles({"memory": "p1:m2"})
    probe = service.get_memory_settings()["format_probe"]

    assert probe["status"] == "passed"
    assert probe["provider_ref"] == "p1:m1"
    assert probe["stale"] is True
    assert probe["stale_reason"] == "memory_role_changed"


def test_post_turn_memory_refresh_gate_blocks_failed_or_stale_probe(tmp_path):
    service = _service(config_path=tmp_path / "config.json")
    session_id = service.store.create_session(title="Probe gate")
    request = RunRequest(query="q", session_id=session_id, workspace_key="default", user_key="default")
    outcome = SimpleNamespace(status="completed", run_id="run-1", answer=SimpleNamespace(text="Usable terminal output."))

    service.update_memory_settings(
        {
            "format_probe": {
                "status": "failed",
                "provider_ref": "p1:m1",
                "capability": "json_schema_strict",
                "structured_strategy": "json_schema_strict",
                "checked_at": 1,
                "error_code": "JSON_PARSE_FAILED",
                "diagnostics": {"response_shape": "prose"},
            }
        }
    )
    assert service._should_refresh_memory_after_run(request, outcome) is False

    service.update_memory_settings(
        {
            "format_probe": {
                "status": "passed",
                "provider_ref": "p1:m1",
                "capability": "json_schema_strict",
                "structured_strategy": "json_schema_strict",
                "checked_at": 1,
                "stale": True,
                "stale_reason": "memory_provider_changed",
                "diagnostics": {"response_shape": "json_object"},
            }
        }
    )
    assert service._should_refresh_memory_after_run(request, outcome) is False


def test_post_turn_memory_refresh_toggle_controls_enqueue_gate():
    service = _service()
    session_id = service.store.create_session(title="Toggle")
    request = RunRequest(query="q", session_id=session_id, workspace_key="default", user_key="default")
    outcome = SimpleNamespace(
        status="completed",
        run_id="run-1",
        answer=SimpleNamespace(text="Usable terminal output."),
    )

    assert service._should_refresh_memory_after_run(request, outcome) is True
    service.update_memory_settings({"post_turn_memory_refresh_enabled": False})

    assert service._should_refresh_memory_after_run(request, outcome) is False


def test_memory_format_probe_keeps_thinking_disabled_when_setting_enabled(tmp_path):
    from fastapi.testclient import TestClient

    llm = _ProbeKwargsLLM()
    service = _service(config_path=tmp_path / "config.json", memory_llm=llm)
    service.update_memory_settings({"memory_thinking_enabled": True})
    client = TestClient(create_app(service))

    probe = client.post("/api/v1/settings/memory/format-probe").json()

    assert probe["status"] == "passed"
    assert llm.kwargs
    assert all(call["thinking_budget"] == 0 for call in llm.kwargs)


class _ProbeKwargsLLM(FakeLLM):
    def __init__(self) -> None:
        super().__init__()
        self.kwargs: list[dict] = []

    def complete(self, messages, **kwargs):  # type: ignore[override]
        self.calls.append(list(messages))
        self.kwargs.append(kwargs)
        return SimpleNamespace(text='{"summary":"Probe ok."}', finish_reason="stop", model=self.model)
