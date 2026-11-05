"""T50.1 / DR-ITER-147 External search provider state contracts."""

from __future__ import annotations

import json
import time
import urllib.error
import threading
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from io import BytesIO
from pathlib import Path

import pytest

from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.search_provider_state import derive_provider_state
from deepresearch_core.baseline.search_usage import resolve_search_usage
from deepresearch_core.baseline.service import BaselineService
from deepresearch_core.baseline.budget import RunRequest
from deepresearch_core.baseline.adapters import ResilientSearchAdapter
from deepresearch_core.config import (
    commit_search_provider_config,
    load_provider_config,
    load_search_provider_state_config,
    save_non_search_provider_config,
)
from deepresearch_core.connectivity import check_named_search_provider_connectivity
from deepresearch_core.settings import ProviderSettings


class _Response:
    def __init__(self, body: dict, *, status: int = 200, headers: Message | None = None) -> None:
        self.status = status
        self.headers = headers or Message()
        self._body = BytesIO(json.dumps(body).encode("utf-8"))

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def close(self) -> None:
        return None


def _valid_body(url: str) -> dict:
    if "brave.com" in url:
        return {"web": {"results": []}}
    if "tavily.com" in url or "exa.ai" in url:
        return {"results": []}
    return {"success": True, "data": []}


def _success_opener(request, timeout):  # noqa: ANN001, ANN202
    assert timeout <= 10.0
    return _Response(_valid_body(request.full_url))


def _write_config(path: Path, provider: dict, state: dict | None = None) -> None:
    payload: dict = {"provider": provider}
    if state is not None:
        payload["search_provider_state"] = state
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_legacy_key_is_not_tested_yellow_and_runtime_eligible() -> None:
    state = derive_provider_state(
        "brave",
        configured=True,
        credential_source="config",
        saved={"enabled": True},
        now=100.0,
    )
    assert state.probe_state == "not_tested"
    assert state.lamp == "yellow"
    assert state.runtime_eligible and state.selectable
    resolved = resolve_search_usage(
        {"usage_mode": "auto", "fallback_chain": ["brave"]},
        source="global",
        configured={
            "brave": {
                **state.to_dict(),
                "search_ready": True,
                "extract_ready": False,
            }
        },
    ).to_dict()
    assert resolved["credential_revision_snapshot"] == {"brave": 0}


def test_available_probe_stales_after_24_hours() -> None:
    state = derive_provider_state(
        "exa",
        configured=True,
        credential_source="config",
        saved={
            "enabled": True,
            "credential_revision": 2,
            "last_probe": {
                "probe_state": "available",
                "credential_state": "accepted",
                "availability": "available",
                "checked_at": 10.0,
                "probe_credential_revision": 2,
            },
        },
        now=10.0 + 86400.0,
    )
    assert state.probe_state == "stale"
    assert state.runtime_eligible


@pytest.mark.parametrize(
    ("code", "body", "probe_state", "credential_state"),
    [
        (401, {"error": "bad key"}, "credential_rejected", "rejected"),
        (403, {"error": "bad key"}, "credential_rejected", "rejected"),
        (403, None, "service_unavailable", "unknown"),
        (402, {"error": "credits"}, "action_required", "accepted"),
        (429, {"error": "rate"}, "rate_limited", "accepted"),
        (503, {"error": "down"}, "service_unavailable", "unknown"),
    ],
)
def test_probe_http_matrix(code: int, body: dict | None, probe_state: str, credential_state: str) -> None:
    headers = Message()
    headers["Retry-After"] = "99999"

    def opener(request, timeout):  # noqa: ANN001, ANN202
        data = json.dumps(body).encode() if body is not None else b"<html>blocked</html>"
        raise urllib.error.HTTPError(
            request.full_url,
            code,
            "failure",
            headers,
            BytesIO(data),
        )

    result = check_named_search_provider_connectivity(
        ProviderSettings(brave_api_key="candidate"),
        provider_id="brave",
        opener=opener,
    )
    assert result["probe_state"] == probe_state
    assert result["credential_state"] == credential_state
    if code == 429:
        assert 1 <= result["retry_after"] - time.time() <= 3601


def test_invalid_2xx_is_soft_failure() -> None:
    result = check_named_search_provider_connectivity(
        ProviderSettings(tavily_api_key="candidate"),
        provider_id="tavily",
        opener=lambda request, timeout: _Response({}),
    )
    assert result["probe_state"] == "service_unavailable"
    assert result["credential_state"] == "unknown"


def test_partial_save_persists_only_accepted_draft(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", config_path=config)

    def opener(request, timeout):  # noqa: ANN001, ANN202
        if "tavily.com" in request.full_url:
            raise urllib.error.HTTPError(
                request.full_url,
                401,
                "unauthorized",
                Message(),
                BytesIO(b'{"error":"bad key"}'),
            )
        return _Response(_valid_body(request.full_url))

    result = service.save_search_providers(
        {
            "expected_revision": 0,
            "providers": {
                "brave": {"api_key": "brave-good"},
                "tavily": {"api_key": "tavily-bad"},
            },
        },
        opener=opener,
    )
    assert result["operation_status"] == "partial"
    assert result["saved"] == ["brave"]
    assert result["rejected"] == ["tavily"]
    stored = load_provider_config(config)
    assert stored.brave_api_key == "brave-good"
    assert stored.tavily_api_key == ""
    raw_config = json.loads(config.read_text(encoding="utf-8"))
    assert set(raw_config["provider"]["brave_api_key"]) == {"secret_ref"}
    assert "brave-good" not in config.read_text(encoding="utf-8")
    assert "brave-good" not in json.dumps(raw_config["search_provider_state"])
    assert "brave-good" not in json.dumps(result)
    assert "tavily-bad" not in json.dumps(result)
    service.store.close()


def test_failed_k2_draft_does_not_pollute_stored_k1_state(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    _write_config(
        config,
        {"brave_api_key": "K1"},
        {
            "revision": 4,
            "providers": {
                "brave": {
                    "enabled": True,
                    "credential_revision": 7,
                    "last_probe": {
                        "probe_state": "available",
                        "credential_state": "accepted",
                        "availability": "available",
                        "checked_at": time.time(),
                        "probe_credential_revision": 7,
                    },
                }
            },
        },
    )
    service = BaselineService(
        provider=ProviderSettings(brave_api_key="K1"),
        db_path=":memory:",
        config_path=config,
        credential_sources={"brave": "config"},
    )

    def reject(request, timeout):  # noqa: ANN001, ANN202
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "unauthorized",
            Message(),
            BytesIO(b'{"error":"bad key"}'),
        )

    result = service.save_search_providers(
        {"expected_revision": 4, "providers": {"brave": {"api_key": "K2"}}},
        opener=reject,
    )
    assert result["operation_status"] == "rejected"
    assert result["revision"] == 4
    assert load_provider_config(config).brave_api_key == "K1"
    stored_state = load_search_provider_state_config(config)
    assert stored_state["providers"]["brave"]["credential_revision"] == 7
    assert stored_state["providers"]["brave"]["last_probe"]["probe_state"] == "available"
    service.store.close()


def test_save_cas_conflict_is_zero_persistence(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", config_path=config)
    service.save_search_providers(
        {"expected_revision": 0, "providers": {"exa": {"api_key": "first"}}},
        opener=_success_opener,
    )
    with pytest.raises(BaselineError) as exc:
        service.save_search_providers(
            {"expected_revision": 0, "providers": {"brave": {"api_key": "stale"}}},
            opener=_success_opener,
        )
    assert exc.value.code is ErrorCode.SEARCH_PROVIDER_STATE_CONFLICT
    assert load_provider_config(config).brave_api_key == ""
    service.store.close()


def test_env_credential_is_read_only_and_never_copied_to_config(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    service = BaselineService(
        provider=ProviderSettings(exa_api_key="env-secret"),
        db_path=":memory:",
        config_path=config,
        credential_sources={"exa": "env"},
    )
    with pytest.raises(BaselineError) as exc:
        service.save_search_providers(
            {"expected_revision": 0, "providers": {"exa": {"api_key": "draft"}}},
            opener=_success_opener,
        )
    assert exc.value.code is ErrorCode.CREDENTIAL_SOURCE_READ_ONLY
    assert not config.exists() or "env-secret" not in config.read_text(encoding="utf-8")
    service.store.close()


def test_toggle_is_immediate_for_new_resolution_but_does_not_rewrite_policy(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    _write_config(config, {"brave_api_key": "K1"})
    service = BaselineService(
        provider=ProviderSettings(brave_api_key="K1"),
        db_path=":memory:",
        config_path=config,
    )
    before = service.provider_status()
    result = service.toggle_search_provider(
        {
            "expected_revision": before["provider_config_revision"],
            "provider": "brave",
            "enabled": False,
        }
    )
    assert result["providers"]["brave"]["lamp"] == "gray"
    assert not result["providers"]["brave"]["runtime_eligible"]
    policy = service.get_search_usage()
    assert "brave" in policy["fallback_chain"]
    service.store.close()


def test_external_unavailable_does_not_claim_local_or_memory_unavailable() -> None:
    resolved = resolve_search_usage(
        {"usage_mode": "auto", "fallback_chain": ["brave"]},
        source="global",
        configured={
            "brave": {
                "configured": True,
                "enabled": False,
                "runtime_eligible": False,
                "probe_state": "available",
            }
        },
    ).to_dict()
    assert resolved["search_chain"] == []
    assert resolved["external_search_status"]["status"] == "unavailable"
    assert "local" not in json.dumps(resolved).lower()
    assert "memory" not in json.dumps(resolved).lower()


def test_mixed_legacy_put_is_rejected(tmp_path: Path) -> None:
    fastapi = pytest.importorskip("fastapi.testclient")
    from deepresearch_core.baseline import create_app

    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", config_path=tmp_path / "config.json")
    with fastapi.TestClient(create_app(service)) as client:
        response = client.put(
            "/api/v1/providers",
            json={"llm_model": "m", "brave_api_key": "search"},
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "INVALID_REQUEST"
    service.store.close()


def test_transient_test_does_not_change_persisted_or_runtime_state(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", config_path=config)
    before = service.provider_status()
    result = service.test_search_tools(
        {"provider": "exa", "exa_api_key": "candidate-only"},
        opener=_success_opener,
    )
    after = service.provider_status()
    assert result["probe_state"] == "available"
    assert after == before
    assert not config.exists()
    service.store.close()


def test_http_save_and_toggle_use_new_contract_without_vendor_egress(tmp_path: Path, monkeypatch) -> None:
    fastapi = pytest.importorskip("fastapi.testclient")
    from deepresearch_core.baseline import create_app
    from deepresearch_core.baseline import service as service_module

    def fake_probe(provider, *, provider_id, timeout_s, opener=None):  # noqa: ANN001, ANN202, ARG001
        return {
            "provider": provider_id,
            "ok": True,
            "status": "reachable",
            "probe_state": "available",
            "credential_state": "accepted",
            "availability": "available",
            "retry_after": None,
            "reason_codes": [],
            "error_message": "",
        }

    monkeypatch.setattr(service_module, "check_named_search_provider_connectivity", fake_probe)
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", config_path=tmp_path / "config.json")
    with fastapi.TestClient(create_app(service)) as client:
        saved = client.post(
            "/api/v1/settings/search-providers/save",
            json={"expected_revision": 0, "providers": {"exa": {"api_key": "candidate"}}},
        )
        assert saved.status_code == 200
        assert saved.json()["providers"]["exa"]["probe_state"] == "available"
        toggled = client.patch(
            "/api/v1/settings/search-providers/enabled",
            json={"expected_revision": 1, "provider": "exa", "enabled": False},
        )
        assert toggled.status_code == 200
        assert toggled.json()["providers"]["exa"]["enabled"] is False
    service.store.close()


def test_worker_revision_mismatch_skips_only_external_provider(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    _write_config(
        config,
        {"brave_api_key": "current"},
        {
            "revision": 2,
            "providers": {
                "brave": {
                    "enabled": True,
                    "credential_revision": 2,
                    "last_probe": {
                        "probe_state": "available",
                        "credential_state": "accepted",
                        "availability": "available",
                        "checked_at": time.time(),
                        "probe_credential_revision": 2,
                    },
                }
            },
        },
    )
    service = BaselineService(
        provider=ProviderSettings(brave_api_key="current"),
        db_path=":memory:",
        config_path=config,
        credential_sources={"brave": "config"},
    )
    request = RunRequest.from_payload(
        {
            "query": "revision guard",
            "mode": "fast",
            "resolved_search_usage": {
                "source": "global",
                "effective_policy": {"usage_mode": "auto", "fallback_chain": ["brave"]},
                "search_chain": ["brave"],
                "extract_chain": [],
                "credential_revision_snapshot": {"brave": 1},
                "reason_codes": [],
            },
        }
    )
    guarded = service.enforce_worker_search_credential_snapshot(request)
    assert guarded.resolved_search_usage["search_chain"] == []
    assert "brave_provider_credential_changed" in guarded.resolved_search_usage["reason_codes"]
    service.store.close()


def test_empty_frozen_external_chain_never_calls_live_provider() -> None:
    class _Client:
        settings = None

        def __getattr__(self, name):  # noqa: ANN001, ANN204
            if name.endswith("search") or name == "search":
                raise AssertionError("live External provider must not be called")
            raise AttributeError(name)

    class _Store:
        def cache_get(self, key):  # noqa: ANN001, ANN201
            return None

    class _Retriever:
        def retrieve(self, query, top_k=5):  # noqa: ANN001, ANN201, ARG002
            return []

        def search(self, query, top_k=5):  # noqa: ANN001, ANN201, ARG002
            return []

    adapter = ResilientSearchAdapter(_Client(), store=_Store(), retriever=_Retriever())
    result = adapter.search_with_outcome(
        "internal evidence can continue",
        resolved_usage={"search_chain": [], "extract_chain": [], "reason_codes": ["brave_disabled"]},
    )
    assert result.status["providers_attempted"] == []
    assert result.status["resolved_search_chain"] == []


def test_effective_loader_reports_env_ownership_over_plaintext_config(tmp_path: Path, monkeypatch) -> None:
    import run_api

    config = tmp_path / "config.json"
    _write_config(config, {"brave_api_key": "config-key"})
    monkeypatch.setattr(run_api, "default_config_path", lambda: config)
    monkeypatch.setenv("DEEPRESEARCH_BRAVE_API_KEY", "env-key")
    settings, sources = run_api._load_settings_with_sources()
    assert settings.brave_api_key == "env-key"
    assert sources["brave"] == "env"
    assert load_provider_config(config).brave_api_key == "config-key"


def test_disabled_provider_without_draft_is_not_probed(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    _write_config(
        config,
        {"brave_api_key": "stored"},
        {"revision": 1, "providers": {"brave": {"enabled": False, "credential_revision": 1}}},
    )
    service = BaselineService(
        provider=ProviderSettings(brave_api_key="stored"),
        db_path=":memory:",
        config_path=config,
    )

    def forbidden_opener(request, timeout):  # noqa: ANN001, ANN202
        raise AssertionError("disabled unchanged provider must not be probed")

    result = service.save_search_providers(
        {"expected_revision": 1, "providers": {}},
        opener=forbidden_opener,
    )
    assert result["operation_status"] == "noop"
    assert result["revision"] == 1
    service.store.close()


def test_save_toggle_race_uses_cas_and_never_half_writes(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", config_path=config)
    entered = threading.Event()
    release = threading.Event()

    def blocked_opener(request, timeout):  # noqa: ANN001, ANN202
        entered.set()
        assert release.wait(timeout=5)
        return _Response(_valid_body(request.full_url))

    with ThreadPoolExecutor(max_workers=2) as executor:
        pending_save = executor.submit(
            service.save_search_providers,
            {"expected_revision": 0, "providers": {"brave": {"api_key": "candidate"}}},
            opener=blocked_opener,
        )
        assert entered.wait(timeout=5)
        toggled = service.toggle_search_provider(
            {"expected_revision": 0, "provider": "brave", "enabled": False}
        )
        release.set()
        with pytest.raises(BaselineError) as exc:
            pending_save.result(timeout=10)
    assert toggled["revision"] == 1
    assert exc.value.code is ErrorCode.SEARCH_PROVIDER_STATE_CONFLICT
    assert load_provider_config(config).brave_api_key == ""
    assert load_search_provider_state_config(config)["providers"]["brave"]["enabled"] is False
    service.store.close()


def test_search_provider_commit_failure_leaves_runtime_unchanged(tmp_path: Path, monkeypatch) -> None:
    from deepresearch_core.baseline import service as service_module

    config = tmp_path / "config.json"
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:", config_path=config)
    before = service.provider_status()

    def fail_commit(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise OSError("disk unavailable")

    monkeypatch.setattr(service_module, "commit_search_provider_config", fail_commit)
    with pytest.raises(OSError, match="disk unavailable"):
        service.save_search_providers(
            {"expected_revision": 0, "providers": {"exa": {"api_key": "candidate"}}},
            opener=_success_opener,
        )
    assert service.provider_status() == before
    assert service._provider.exa_api_key == ""  # noqa: SLF001
    service.store.close()


def test_worker_refreshes_probe_state_even_when_credentials_are_unchanged(monkeypatch) -> None:
    from deepresearch_core.baseline import worker

    calls: list[str] = []

    class _WorkerService:
        def reconfigure(self, provider, **kwargs):  # noqa: ANN001, ANN003
            calls.append("reconfigure")

        def reload_search_provider_state_for_worker(self) -> None:
            calls.append("reload_state")

        def enforce_worker_search_credential_snapshot(self, request):  # noqa: ANN001, ANN201
            return request

        def execute_run(self, request, run_id):  # noqa: ANN001, ANN202
            calls.append(f"execute:{run_id}")

    settings = ProviderSettings(brave_api_key="same-key")
    request = RunRequest.from_payload(
        {
            "query": "worker readiness refresh",
            "mode": "fast",
            "deadline_at": time.time() + 60,
            "resolved_search_usage": {
                "source": "global",
                "effective_policy": {"usage_mode": "auto", "fallback_chain": ["brave"]},
                "search_chain": ["brave"],
                "extract_chain": [],
                "reason_codes": [],
            },
        }
    )
    monkeypatch.setattr(worker, "_service", _WorkerService())
    monkeypatch.setattr(worker, "_last_provider_settings", None)
    monkeypatch.setattr(worker, "_last_provider_sources", None)
    monkeypatch.setattr(
        worker,
        "_load_worker_provider_settings",
        lambda: (settings, {"brave": "config"}),
    )

    worker.run_job(request.to_payload(), "run-one")
    worker.run_job(request.to_payload(), "run-two")

    assert calls == ["reconfigure", "execute:run-one", "reload_state", "execute:run-two"]


def test_non_search_save_cannot_overwrite_a_concurrent_search_credential(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    _write_config(config, {"brave_api_key": "old", "llm_model": "old-model"})
    initial = load_search_provider_state_config(config)
    committed = commit_search_provider_config(
        config,
        expected_revision=0,
        provider_updates={"brave_api_key": "new"},
        provider_states=initial["providers"],
    )
    assert committed["revision"] == 1

    save_non_search_provider_config(
        config,
        ProviderSettings(brave_api_key="old", llm_model="new-model"),
    )

    stored = load_provider_config(config)
    assert stored.brave_api_key == "new"
    assert stored.llm_model == "new-model"
    assert load_search_provider_state_config(config)["revision"] == 1


def test_search_save_runtime_merge_preserves_concurrent_non_search_update(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    _write_config(config, {"brave_api_key": "old", "llm_model": "old-model"})
    service = BaselineService(
        provider=ProviderSettings(brave_api_key="old", llm_model="old-model"),
        db_path=":memory:",
        config_path=config,
        credential_sources={"brave": "config"},
    )
    entered = threading.Event()
    release = threading.Event()

    def blocking_opener(request, timeout):  # noqa: ANN001, ANN202, ARG001
        entered.set()
        assert release.wait(timeout=5)
        return _Response(_valid_body(request.full_url))

    with ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(
            service.save_search_providers,
            {"expected_revision": 0, "providers": {"brave": {"api_key": "new"}}},
            opener=blocking_opener,
        )
        assert entered.wait(timeout=5)
        service.update_providers({"llm_model": "new-model"})
        release.set()
        assert pending.result(timeout=10)["operation_status"] == "success"

    assert service._provider.brave_api_key == "new"  # noqa: SLF001
    assert service._provider.llm_model == "new-model"  # noqa: SLF001
    stored = load_provider_config(config)
    assert stored.brave_api_key == "new"
    assert stored.llm_model == "new-model"
    service.store.close()


@pytest.mark.parametrize(
    "payload",
    [
        {"expected_revision": True, "providers": {}},
        {"expected_revision": -1, "providers": {}},
        {"expected_revision": 0, "providers": {"brave": None}},
        {"expected_revision": 0, "providers": {"brave": {"remove": "false"}}},
        {"expected_revision": 0, "providers": {"brave": {"api_key": 123}}},
    ],
)
def test_save_rejects_ambiguous_state_mutation_types(tmp_path: Path, payload: dict) -> None:
    service = BaselineService(
        provider=ProviderSettings(),
        db_path=":memory:",
        config_path=tmp_path / "config.json",
    )
    with pytest.raises(BaselineError) as exc:
        service.save_search_providers(payload, opener=_success_opener)
    assert exc.value.code is ErrorCode.INVALID_REQUEST
    service.store.close()


def test_toggle_requires_a_boolean_enabled_value(tmp_path: Path) -> None:
    service = BaselineService(
        provider=ProviderSettings(),
        db_path=":memory:",
        config_path=tmp_path / "config.json",
    )
    with pytest.raises(BaselineError) as exc:
        service.toggle_search_provider(
            {"expected_revision": 0, "provider": "brave", "enabled": "false"}
        )
    assert exc.value.code is ErrorCode.INVALID_REQUEST
    service.store.close()
