"""Runtime BYOK provider configuration: update_providers + reconfigure + persistence."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from deepresearch_core.baseline import BaselineService
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.config import load_provider_config
from deepresearch_core.settings import ProviderSettings


VALID_PAYLOAD = {
    "llm_base_url": "https://llm.example.test/v1",
    "llm_api_key": "sk-test-123",
    "llm_model": "gpt-test",
}


def test_update_providers_flips_ready_and_status():
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    assert service.provider_status()["has_llm_key"] is False

    status = service.update_providers(VALID_PAYLOAD)

    assert status["has_llm_key"] is True
    assert status["brave_ready"] is False
    assert status["firecrawl_ready"] is False
    assert status["llm_model"] == "gpt-test"
    # Secret-free: no raw keys leak in the returned status.
    assert "llm_api_key" not in status
    assert "llm_api_key" not in status
    assert "sk-test-123" not in str(status)


def test_update_providers_enables_runs():
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    service.update_providers(VALID_PAYLOAD)
    # With a configured LLM, a run can at least be created (not PROVIDER_NOT_CONFIGURED).
    result = service.create_run({"query": "hello world", "mode": "fast"}, background=True)
    assert result["run_id"]
    assert result["status"] == "running"


def test_update_providers_persists_to_config_file(tmp_path):
    config_path = tmp_path / "config.json"
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", config_path=config_path)

    service.update_providers(VALID_PAYLOAD)

    assert config_path.exists()
    reloaded = load_provider_config(config_path)
    assert reloaded.llm_base_url == "https://llm.example.test/v1"
    assert reloaded.llm_api_key == "sk-test-123"
    assert reloaded.llm_model == "gpt-test"
    assert reloaded.brave_api_key == ""


def test_update_providers_merges_partial_updates():
    service = BaselineService(provider=ProviderSettings(**VALID_PAYLOAD), db_path=":memory:")
    # A partial update (only the model) keeps existing keys.
    status = service.update_providers({"llm_model": "gpt-test-2"})
    assert status["llm_model"] == "gpt-test-2"
    assert status["has_llm_key"] is True
    assert status["brave_ready"] is False


def test_update_providers_rejects_non_dict():
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    with pytest.raises(BaselineError) as exc:
        service.update_providers(["not", "a", "dict"])  # type: ignore[arg-type]
    assert exc.value.code is ErrorCode.INVALID_REQUEST


def test_update_providers_ignores_unknown_fields():
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    payload = dict(VALID_PAYLOAD)
    payload["totally_unknown"] = "x"
    status = service.update_providers(payload)
    assert status["has_llm_key"] is True


def test_provider_config_save_failure_leaves_runtime_unchanged(tmp_path, monkeypatch):
    from deepresearch_core.baseline import service as service_module

    service = BaselineService(
        provider=ProviderSettings(llm_model="existing-model"),
        db_path=":memory:",
        config_path=tmp_path / "config.json",
    )
    before = service.provider_status()

    def _fail_save(*_args, **_kwargs):
        raise OSError("disk unavailable")

    monkeypatch.setattr(service_module, "save_non_search_provider_config", _fail_save)

    with pytest.raises(OSError, match="disk unavailable"):
        service.update_providers({"llm_model": "new-model"})

    assert service.provider_status() == before
    assert service._provider.llm_model == "existing-model"  # noqa: SLF001 - runtime atomicity assertion
    service.store.close()


def test_concurrent_partial_provider_updates_are_serialized(tmp_path, monkeypatch):
    from deepresearch_core.baseline import service as service_module

    service = BaselineService(
        provider=ProviderSettings(),
        db_path=":memory:",
        config_path=tmp_path / "config.json",
    )
    first_merge_started = threading.Event()
    release_first_merge = threading.Event()
    original_merge = service_module.merge_provider_config

    def _delayed_merge(current, updates):  # noqa: ANN001, ANN202
        merged = original_merge(current, updates)
        if "llm_base_url" in updates:
            first_merge_started.set()
            assert release_first_merge.wait(timeout=5)
        return merged

    monkeypatch.setattr(service_module, "merge_provider_config", _delayed_merge)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.update_providers, {"llm_base_url": "https://one.example/v1"})
        assert first_merge_started.wait(timeout=5)
        second = executor.submit(service.update_providers, {"llm_model": "model-two"})
        time.sleep(0.05)
        assert second.done() is False
        release_first_merge.set()
        first.result(timeout=10)
        second.result(timeout=10)

    assert service._provider.llm_base_url == "https://one.example/v1"  # noqa: SLF001
    assert service._provider.llm_model == "model-two"  # noqa: SLF001
    persisted = load_provider_config(tmp_path / "config.json")
    assert persisted.llm_base_url == "https://one.example/v1"
    assert persisted.llm_model == "model-two"
    service.store.close()


# --- HTTP surface ------------------------------------------------------------

fastapi_testclient = pytest.importorskip("fastapi.testclient", reason="fastapi not installed")


def _client():
    from fastapi.testclient import TestClient

    from deepresearch_core.baseline import create_app

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    return TestClient(create_app(service))


def test_put_providers_then_get_is_ready():
    client = _client()
    assert client.get("/api/v1/providers").json()["has_llm_key"] is False

    res = client.put("/api/v1/providers", json=VALID_PAYLOAD)
    assert res.status_code == 200
    assert res.json()["has_llm_key"] is True

    got = client.get("/api/v1/providers").json()
    assert got["has_llm_key"] is True
    assert got["brave_ready"] is False
    assert "llm_api_key" not in got
    # Regression guard: no configured key string ever appears in the status response.
    blob = str(got)
    assert "sk-test-123" not in blob


def test_put_providers_invalid_payload_returns_400():
    client = _client()
    res = client.put("/api/v1/providers", json=["nope"])
    assert res.status_code == 400
    assert res.json()["error"]["code"] == ErrorCode.INVALID_REQUEST.value
