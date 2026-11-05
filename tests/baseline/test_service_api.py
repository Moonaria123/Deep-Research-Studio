"""BaselineService facade + FastAPI error envelope and endpoints."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.settings import ProviderSettings


@pytest.fixture
def unconfigured_service() -> BaselineService:
    # No provider keys => LLM is unconfigured but the service still serves status/sessions.
    return BaselineService(provider=ProviderSettings(), db_path=":memory:")


def test_provider_status_is_secret_free(unconfigured_service):
    status = unconfigured_service.provider_status()
    assert status["has_llm_key"] is False
    assert status["brave_ready"] is False
    # No raw secret fields are exposed.
    assert "llm_api_key" not in status


def test_create_run_without_llm_raises_provider_not_configured(unconfigured_service):
    with pytest.raises(BaselineError) as exc:
        unconfigured_service.create_run({"query": "hello", "mode": "fast"}, background=False)
    assert exc.value.code is ErrorCode.PROVIDER_NOT_CONFIGURED


def test_get_missing_run_raises_run_not_found(unconfigured_service):
    with pytest.raises(BaselineError) as exc:
        unconfigured_service.get_run("does-not-exist")
    assert exc.value.code is ErrorCode.RUN_NOT_FOUND


def test_sessions_roundtrip(unconfigured_service):
    created = unconfigured_service.create_session(title="My research")
    assert created["session_id"]
    sessions = unconfigured_service.list_sessions()
    assert any(s["id"] == created["session_id"] for s in sessions)


def test_patch_session_updates_pin_and_mode_default(unconfigured_service):
    created = unconfigured_service.create_session(title="Pinned research")

    result = unconfigured_service.patch_session(
        created["session_id"],
        {"pinned": True, "mode_default": "deep"},
    )

    assert result["session"]["pinned"] is True
    assert result["session"]["mode_default"] == "deep"


def test_list_sessions_orders_pinned_before_recent(unconfigured_service):
    first = unconfigured_service.create_session(title="Recent")
    second = unconfigured_service.create_session(title="Pinned")

    unconfigured_service.patch_session(second["session_id"], {"pinned": True})
    sessions = unconfigured_service.list_sessions()

    assert sessions[0]["id"] == second["session_id"]
    assert any(session["id"] == first["session_id"] for session in sessions)


def test_create_run_uses_session_mode_default_when_mode_missing(unconfigured_service, monkeypatch):
    created = unconfigured_service.create_session(title="Default mode")
    unconfigured_service.patch_session(created["session_id"], {"mode_default": "multiple"})

    class CaptureBackend:
        def __init__(self):
            self.request = None

        def enqueue(self, request, run_id):  # noqa: ANN001
            self.request = request

        def health(self):
            return {"job_backend": "capture"}

    backend = CaptureBackend()
    monkeypatch.setattr(unconfigured_service, "_llm_ready", True)
    monkeypatch.setattr(unconfigured_service, "_job_backend", backend)

    result = unconfigured_service.create_run(
        {"query": "Use the saved default", "session_id": created["session_id"]},
        background=True,
    )

    run = unconfigured_service.store.get_run(result["run_id"])
    assert run["mode"] == "multiple"
    assert backend.request.mode.value == "multiple"


# --- HTTP surface ------------------------------------------------------------

fastapi_testclient = pytest.importorskip("fastapi.testclient", reason="fastapi not installed")


@pytest.fixture
def client(unconfigured_service):
    from fastapi.testclient import TestClient

    return TestClient(create_app(unconfigured_service))


def test_health_endpoint(client):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok" or "fts5" in res.json()


def test_error_envelope_shape_on_missing_run(client):
    res = client.get("/api/v1/runs/nope")
    assert res.status_code == 404
    body = res.json()
    assert body["error"]["code"] == ErrorCode.RUN_NOT_FOUND.value
    assert "message" in body["error"]


def test_create_run_http_conflict_when_unconfigured(client):
    res = client.post("/api/v1/runs", json={"query": "hello", "mode": "fast"})
    assert res.status_code == 409
    assert res.json()["error"]["code"] == ErrorCode.PROVIDER_NOT_CONFIGURED.value


def test_invalid_mode_returns_400(client):
    res = client.post("/api/v1/runs", json={"query": "hello", "mode": "turbo"})
    assert res.status_code == 400
    assert res.json()["error"]["code"] == ErrorCode.INVALID_REQUEST.value


def test_patch_session_http_roundtrip(client, unconfigured_service):
    session_id = unconfigured_service.create_session(title="HTTP session")["session_id"]

    res = client.patch(f"/api/v1/sessions/{session_id}", json={"pinned": True, "mode_default": "council"})

    assert res.status_code == 200
    body = res.json()
    assert body["session_id"] == session_id
    assert body["session"]["pinned"] is True
    assert body["session"]["mode_default"] == "council"


# --- Council chair assignment ------------------------------------------------

from deepresearch_core.baseline.budget import RunRequest  # noqa: E402
from deepresearch_core.baseline.council import CouncilMemberSpec  # noqa: E402


class _StubAdapter:
    """Minimal LLMAdapter stand-in identified by name."""

    def __init__(self, name: str) -> None:
        self.name = name


def _council_request(**over) -> RunRequest:
    payload = {"query": "What is the impact of X?", "mode": "council"}
    payload.update(over)
    return RunRequest.from_payload(payload)


def test_assign_council_round_robins_candidates(unconfigured_service, monkeypatch):
    a, b = _StubAdapter("a"), _StubAdapter("b")
    monkeypatch.setattr(unconfigured_service, "_resolve_council_members", lambda refs: [a, b])
    monkeypatch.setattr(
        unconfigured_service, "_chair_perspectives", lambda query, size: ["p1", "p2", "p3"]
    )
    specs = unconfigured_service._assign_council(_council_request(council_models=["a", "b"], council_size=3))
    assert [s.adapter for s in specs] == [a, b, a]
    assert [s.perspective for s in specs] == ["p1", "p2", "p3"]
    assert all(isinstance(s, CouncilMemberSpec) for s in specs)


def test_assign_council_single_model_multiple_perspectives(unconfigured_service, monkeypatch):
    only = _StubAdapter("solo")
    monkeypatch.setattr(unconfigured_service, "_resolve_council_members", lambda refs: [only])
    monkeypatch.setattr(unconfigured_service, "_chair_perspectives", lambda query, size: ["x", "y"])
    specs = unconfigured_service._assign_council(_council_request(council_models=["solo"], council_size=2))
    assert [s.adapter for s in specs] == [only, only]
    assert [s.perspective for s in specs] == ["x", "y"]


def test_assign_council_falls_back_to_builtin_lenses_when_chair_fails(unconfigured_service, monkeypatch):
    a = _StubAdapter("a")
    monkeypatch.setattr(unconfigured_service, "_resolve_council_members", lambda refs: [a])
    monkeypatch.setattr(unconfigured_service, "_chair_perspectives", lambda query, size: [])
    specs = unconfigured_service._assign_council(_council_request(council_models=["a"], council_size=3))
    perspectives = [s.perspective for s in specs]
    assert len(specs) == 3
    assert len(set(perspectives)) == 3  # built-in lenses are distinct
    assert all(p for p in perspectives)


def test_memories_endpoint_accepts_query_param(client):
    res = client.get("/api/v1/memories", params={"query": "anything", "workspace_key": "default"})
    assert res.status_code == 200
    assert "memories" in res.json()


def test_history_search_missing_session_returns_404(client):
    res = client.get("/api/v1/sessions/ghost/history/search", params={"query": "x"})
    assert res.status_code == 404


def test_history_search_requires_query(client, unconfigured_service):
    sid = unconfigured_service.create_session(title="t")["session_id"]
    res = client.get(f"/api/v1/sessions/{sid}/history/search", params={"query": ""})
    assert res.status_code == 400


def test_history_search_returns_results_shape_gracefully(client, unconfigured_service):
    # Unconfigured service has no embedder => graceful empty results, never 500.
    sid = unconfigured_service.create_session(title="t")["session_id"]
    res = client.get(f"/api/v1/sessions/{sid}/history/search", params={"query": "x", "scope": "cross"})
    assert res.status_code == 200
    body = res.json()
    assert body["scope"] == "cross" and body["results"] == []


def test_assign_council_empty_candidates_returns_empty(unconfigured_service, monkeypatch):
    # No candidates and an unconfigured default LLM => no members (graceful).
    monkeypatch.setattr(unconfigured_service, "_resolve_council_members", lambda refs: [])
    monkeypatch.setattr(unconfigured_service, "_adapter_for_ref", lambda ref: None)
    monkeypatch.setattr(unconfigured_service, "_resolve_run_llm", lambda request: None)
    specs = unconfigured_service._assign_council(_council_request(council_models=["x"], council_size=3))
    assert specs == []
