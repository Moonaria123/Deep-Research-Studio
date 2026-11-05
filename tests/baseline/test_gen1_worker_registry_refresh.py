"""Regression coverage for process-cached worker LLM registry refresh."""

from __future__ import annotations

import time
from dataclasses import replace
from types import SimpleNamespace

from deepresearch_core.baseline import worker
from deepresearch_core.baseline.budget import RunRequest
from deepresearch_core.providers_registry import (
    LLMProviderProfile,
    LLMRegistry,
    LLMRoles,
    ModelProfile,
)
from deepresearch_core.settings import ProviderSettings


def _request() -> RunRequest:
    return RunRequest.from_payload(
        {
            "query": "worker registry refresh",
            "mode": "fast",
            "deadline_at": time.time() + 60,
        }
    )


def _registry(*, api_key: str = "key-one", enabled: bool = True, default: str = "provider:model-a") -> LLMRegistry:
    return LLMRegistry(
        providers=(
            LLMProviderProfile(
                id="provider",
                base_url="https://provider.invalid/v1",
                api_key=api_key,
                models=(ModelProfile(id="model-a"), ModelProfile(id="model-b")),
                enabled=enabled,
            ),
        ),
        roles=LLMRoles(default=default),
    )


def test_worker_reconfigures_for_key_role_and_enabled_changes_only(monkeypatch) -> None:
    calls: list[str] = []

    class _WorkerService:
        def reconfigure(self, provider, **kwargs) -> None:  # noqa: ANN001, ANN003
            calls.append("reconfigure")

        def reload_search_provider_state_for_worker(self) -> None:
            calls.append("reload_state")

        def enforce_worker_search_credential_snapshot(self, request):  # noqa: ANN001, ANN201
            return request

        def execute_run(self, request, run_id: str) -> None:  # noqa: ANN001
            calls.append(f"execute:{run_id}")

    initial = _registry()
    key_replaced = _registry(api_key="key-two")
    role_changed = replace(key_replaced, roles=LLMRoles(default="provider:model-b"))
    provider_disabled = replace(
        role_changed,
        providers=(replace(role_changed.providers[0], enabled=False),),
    )
    snapshots = iter((initial, key_replaced, role_changed, provider_disabled, provider_disabled))
    settings = ProviderSettings(brave_api_key="same-search-key")

    monkeypatch.setattr(worker, "_service", _WorkerService())
    monkeypatch.setattr(worker, "_last_provider_settings", None)
    monkeypatch.setattr(worker, "_last_provider_sources", None)
    monkeypatch.setattr(worker, "_last_llm_registry", None)
    monkeypatch.setattr(
        worker,
        "_load_worker_provider_settings",
        lambda: (settings, {"brave": "config"}, next(snapshots)),
    )

    request = _request()
    for index in range(5):
        worker.run_job(request.to_payload(), f"run-{index}")

    assert calls == [
        "reconfigure",
        "execute:run-0",
        "reconfigure",
        "execute:run-1",
        "reconfigure",
        "execute:run-2",
        "reconfigure",
        "execute:run-3",
        "reload_state",
        "execute:run-4",
    ]


def test_fresh_worker_reuses_built_snapshot_but_observes_later_rotation(monkeypatch) -> None:
    import run_api

    calls: list[str] = []
    settings = ProviderSettings(brave_api_key="search-test-value")
    initial = _registry()
    rotated = _registry(api_key="rotated-test-value")
    service = SimpleNamespace(
        _provider=settings,
        _search_provider_sources={"brave": "config"},
        _registry=initial,
        reconfigure=lambda *args, **kwargs: calls.append("reconfigure"),
        reload_search_provider_state_for_worker=lambda: calls.append("reload_state"),
        execute_run=lambda *args: calls.append("execute"),
    )
    monkeypatch.setattr(run_api, "build_service", lambda: service)
    monkeypatch.setattr(worker, "_service", None)
    monkeypatch.setattr(worker, "_last_provider_settings", None)
    monkeypatch.setattr(worker, "_last_provider_sources", None)
    monkeypatch.setattr(worker, "_last_llm_registry", None)
    snapshots = iter((initial, rotated))
    monkeypatch.setattr(
        worker, "_load_worker_provider_settings",
        lambda: (settings, {"brave": "config"}, next(snapshots)),
    )

    worker.run_job(_request().to_payload(), "fresh")
    worker.run_job(_request().to_payload(), "rotated")

    assert calls == ["reload_state", "execute", "reconfigure", "execute"]


def test_fresh_worker_observes_rotation_between_build_and_reload(monkeypatch) -> None:
    import run_api

    calls: list[str] = []
    settings = ProviderSettings()
    service = SimpleNamespace(
        _provider=settings,
        _search_provider_sources={},
        _registry=_registry(),
        reconfigure=lambda *args, **kwargs: calls.append("reconfigure"),
        execute_run=lambda *args: calls.append("execute"),
    )
    monkeypatch.setattr(run_api, "build_service", lambda: service)
    monkeypatch.setattr(worker, "_service", None)
    monkeypatch.setattr(worker, "_last_provider_settings", None)
    monkeypatch.setattr(worker, "_last_provider_sources", None)
    monkeypatch.setattr(worker, "_last_llm_registry", None)
    monkeypatch.setattr(
        worker, "_load_worker_provider_settings",
        lambda: (settings, {}, _registry(enabled=False)),
    )

    worker.run_job(_request().to_payload(), "fresh-with-rotation")

    assert calls == ["reconfigure", "execute"]
