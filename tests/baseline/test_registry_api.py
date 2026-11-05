"""Phase 4: LLM registry REST endpoints (CRUD + roles) over the FastAPI app.

Hermetic: no live network. Provider/model fetch + connection tests are covered
elsewhere; here we exercise CRUD, role validation, compat shim, and the
secret-free invariant of every response body.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import deepresearch_core.baseline.service as service_module
from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.secret_vault import SecretVaultError
from deepresearch_core.settings import ProviderSettings

pytest.importorskip("fastapi.testclient", reason="fastapi not installed")


@pytest.fixture
def client(tmp_path: Path):
    from fastapi.testclient import TestClient

    cfg = tmp_path / "config.json"
    svc = BaselineService(provider=ProviderSettings(), db_path=":memory:", config_path=cfg)
    return TestClient(create_app(svc)), cfg


def test_upsert_then_get_registry_is_secret_free(client):
    api, cfg = client
    res = api.put("/api/v1/providers/llm", json={
        "id": "p1", "label": "Prov 1", "type": "openai",
        "base_url": "https://p1.example/v1", "api_key": "sk-secret",
        "models": ["gpt-4o", "gpt-4o-mini"], "enabled": True,
    })
    assert res.status_code == 200
    body = res.json()
    assert "sk-secret" not in json.dumps(body)
    p1 = next(p for p in body["providers"] if p["id"] == "p1")
    assert p1["has_key"] is True and "api_key" not in p1
    assert [m["id"] for m in p1["models"]] == ["gpt-4o", "gpt-4o-mini"]

    got = api.get("/api/v1/providers/llm").json()
    assert any(p["id"] == "p1" for p in got["providers"])
    # The key is persisted only through an encrypted vault reference.
    persisted = cfg.read_text(encoding="utf-8")
    assert "sk-secret" not in persisted
    assert "secret_ref" in persisted


def test_upsert_blank_key_preserves_existing(client):
    api, _ = client
    api.put("/api/v1/providers/llm", json={
        "id": "p1", "type": "openai", "base_url": "https://p1.example/v1",
        "api_key": "sk-secret", "models": ["m"],
    })
    # Toggle enabled without re-sending the key.
    res = api.put("/api/v1/providers/llm", json={"id": "p1", "enabled": False})
    body = res.json()
    p1 = next(p for p in body["providers"] if p["id"] == "p1")
    assert p1["enabled"] is False and p1["has_key"] is True


def test_set_roles_validates_and_persists(client):
    api, _ = client
    api.put("/api/v1/providers/llm", json={
        "id": "p1", "type": "openai", "base_url": "https://p1.example/v1",
        "api_key": "sk", "models": ["gpt-4o"],
    })
    ok = api.put("/api/v1/providers/roles", json={"default": "p1:gpt-4o", "council": ["p1:gpt-4o"]})
    assert ok.status_code == 200
    assert ok.json()["roles"]["default"] == "p1:gpt-4o"

    bad = api.put("/api/v1/providers/roles", json={"default": "ghost:x"})
    assert bad.status_code == 400


def test_delete_provider_scrubs_roles(client):
    api, _ = client
    api.put("/api/v1/providers/llm", json={
        "id": "p1", "type": "openai", "base_url": "https://p1.example/v1",
        "api_key": "sk", "models": ["gpt-4o"],
    })
    api.put("/api/v1/providers/roles", json={"default": "p1:gpt-4o"})
    res = api.delete("/api/v1/providers/llm/p1")
    assert res.status_code == 200
    body = res.json()
    assert all(p["id"] != "p1" for p in body["providers"])
    assert body["roles"]["default"] == ""


def test_upsert_without_id_returns_400(client):
    api, _ = client
    res = api.put("/api/v1/providers/llm", json={"type": "openai"})
    assert res.status_code == 400


def test_upsert_rich_models_roundtrip_secret_free(client):
    api, _ = client
    res = api.put("/api/v1/providers/llm", json={
        "id": "p1", "type": "openai", "base_url": "https://p1.example/v1", "api_key": "sk-x",
        "models": [
            {
                "id": "gpt-4o", "alias": "Omni", "kind": "chat", "context_window": 200000,
                "capabilities": {"vision": True, "ocr": True, "web_search": False, "thinking": True},
                "thinking_budget": 2048, "enabled": True,
            },
            {"id": "text-embed", "kind": "embed", "dimensions": 1536},
        ],
    })
    assert res.status_code == 200
    body = res.json()
    assert "sk-x" not in json.dumps(body)
    p1 = next(p for p in body["providers"] if p["id"] == "p1")
    chat = next(m for m in p1["models"] if m["id"] == "gpt-4o")
    assert chat["alias"] == "Omni" and chat["kind"] == "chat" and chat["context_window"] == 200000
    assert chat["capabilities"]["vision"] is True and chat["capabilities"]["thinking"] is True
    assert chat["thinking_budget"] == 2048
    embed = next(m for m in p1["models"] if m["id"] == "text-embed")
    assert embed["kind"] == "embed" and embed["dimensions"] == 1536


def test_create_run_payload_carries_council_size():
    """POST /runs body flows through RunRequest.from_payload, which parses council_size."""
    from deepresearch_core.baseline.budget import RunRequest

    req = RunRequest.from_payload({"query": "q", "mode": "council", "council_size": 5})
    assert req.council_size == 5


def test_patch_provider_partial_fields_preserve_key(client):
    api, _ = client
    api.put("/api/v1/providers/llm", json={
        "id": "p1", "label": "Old", "type": "openai",
        "base_url": "https://p1.example/v1", "api_key": "sk-secret", "models": ["m"],
    })
    # Patch label + enabled without re-sending the key or models.
    res = api.patch("/api/v1/providers/llm/p1", json={"label": "New", "enabled": False})
    assert res.status_code == 200
    body = res.json()
    p1 = next(p for p in body["providers"] if p["id"] == "p1")
    assert p1["label"] == "New" and p1["enabled"] is False and p1["has_key"] is True
    assert [m["id"] for m in p1["models"]] == ["m"]
    assert "sk-secret" not in json.dumps(body)


def test_patch_provider_unknown_returns_400(client):
    api, _ = client
    res = api.patch("/api/v1/providers/llm/ghost", json={"enabled": True})
    assert res.status_code == 400


def test_saved_key_has_no_reveal_endpoint_and_status_stays_clean(client):
    api, _ = client
    api.put("/api/v1/providers/llm", json={
        "id": "p1", "type": "openai", "base_url": "https://p1.example/v1",
        "api_key": "sk-reveal", "models": ["m"],
    })
    revealed = api.post("/api/v1/providers/llm/p1/reveal-key")
    assert revealed.status_code == 404
    assert "sk-reveal" not in revealed.text
    status = api.get("/api/v1/providers/llm").json()
    assert "sk-reveal" not in json.dumps(status)


def test_per_provider_model_crud_roundtrip(client):
    api, _ = client
    api.put("/api/v1/providers/llm", json={
        "id": "p1", "type": "openai", "base_url": "https://p1.example/v1",
        "api_key": "sk", "models": ["gpt-4o"],
    })
    # Batch add (one new, skip dup id handled as replace).
    add = api.post("/api/v1/providers/llm/p1/models", json={"models": [
        {"id": "gpt-4o-mini", "alias": "Mini"},
        {"id": "text-embed", "kind": "embed", "dimensions": 1536},
    ]})
    assert add.status_code == 200 and add.json()["added"] == 2

    listed = api.get("/api/v1/providers/llm/p1/models").json()
    ids = [m["id"] for m in listed["models"]]
    assert ids == ["gpt-4o", "gpt-4o-mini", "text-embed"]

    # Update one model's alias + capabilities.
    upd = api.patch("/api/v1/providers/llm/p1/models/gpt-4o-mini", json={
        "alias": "Mini2", "capabilities": {"vision": True},
    })
    assert upd.status_code == 200
    mini = next(m for m in upd.json()["models"] if m["id"] == "gpt-4o-mini")
    assert mini["alias"] == "Mini2" and mini["capabilities"]["vision"] is True

    # Delete one model.
    dele = api.delete("/api/v1/providers/llm/p1/models/text-embed")
    assert dele.status_code == 200
    after = next(p for p in dele.json()["providers"] if p["id"] == "p1")
    assert [m["id"] for m in after["models"]] == ["gpt-4o", "gpt-4o-mini"]


def test_delete_model_scrubs_role_binding(client):
    api, _ = client
    api.put("/api/v1/providers/llm", json={
        "id": "p1", "type": "openai", "base_url": "https://p1.example/v1",
        "api_key": "sk", "models": ["gpt-4o", "gpt-4o-mini"],
    })
    api.put("/api/v1/providers/roles", json={"default": "p1:gpt-4o", "council": ["p1:gpt-4o"]})
    res = api.delete("/api/v1/providers/llm/p1/models/gpt-4o")
    assert res.status_code == 200
    body = res.json()
    assert body["roles"]["default"] == ""
    assert body["roles"]["council"] == []


def test_create_model_without_id_returns_400(client):
    api, _ = client
    api.put("/api/v1/providers/llm", json={
        "id": "p1", "type": "openai", "base_url": "https://p1.example/v1", "api_key": "sk", "models": [],
    })
    res = api.post("/api/v1/providers/llm/p1/models", json={"models": [{"alias": "no id"}]})
    assert res.status_code == 400


def test_flat_providers_compat_shim_still_works(client):
    api, _ = client
    # The shipped flat flow must keep working alongside the registry.
    res = api.put("/api/v1/providers", json={
        "llm_base_url": "https://flat.example/v1", "llm_api_key": "sk-flat", "llm_model": "flat-1",
    })
    assert res.status_code == 200
    body = res.json()
    assert body["llm_model"] == "flat-1" and body["has_llm_key"] is True
    assert "sk-flat" not in json.dumps(body)


def test_registry_load_does_not_fallback_when_vault_cannot_decrypt(tmp_path: Path, monkeypatch):
    svc = BaselineService(provider=ProviderSettings(), db_path=":memory:", config_path=None)
    svc._config_path = tmp_path / "config.json"
    monkeypatch.setattr(service_module, "config_has_llm_registry", lambda _path: True)

    def fail_load(*_args, **_kwargs):
        raise SecretVaultError("vault unavailable")

    monkeypatch.setattr(service_module, "load_llm_registry", fail_load)
    with pytest.raises(SecretVaultError, match="vault unavailable"):
        svc._load_or_build_registry()


@pytest.mark.parametrize(
    "mutation",
    [
        "upsert_provider",
        "patch_provider",
        "remove_provider",
        "set_roles",
        "create_model",
        "update_model",
        "delete_model",
    ],
)
def test_registry_persist_failure_keeps_old_runtime_state(tmp_path: Path, monkeypatch, mutation: str):
    svc = BaselineService(provider=ProviderSettings(), db_path=":memory:", config_path=None)
    svc.upsert_llm_provider({
        "id": "p1",
        "type": "openai",
        "base_url": "https://old.example/v1",
        "api_key": "sk-old",
        "models": ["m", "n"],
    })
    svc.set_llm_roles({"default": "p1:m", "memory": "p1:m", "council": ["p1:m"]})
    old_registry = svc._registry
    old_status = old_registry.status()
    old_runtime = svc._llm
    old_memory_settings = svc._memory_settings
    svc._config_path = tmp_path / "config.json"

    def fail_save(*_args, **_kwargs):
        raise SecretVaultError("vault write failed")

    wire_calls: list[bool] = []
    stale_calls: list[str] = []
    monkeypatch.setattr(service_module, "save_llm_registry", fail_save)
    monkeypatch.setattr(svc, "_wire_runtime", lambda: wire_calls.append(True))
    monkeypatch.setattr(svc, "_mark_memory_probe_stale", lambda reason: stale_calls.append(reason))

    def mutate() -> None:
        if mutation == "upsert_provider":
            svc.upsert_llm_provider({"id": "p1", "api_key": "sk-new"})
        elif mutation == "patch_provider":
            svc.patch_llm_provider("p1", {"base_url": "https://new.example/v1", "api_key": "sk-new"})
        elif mutation == "remove_provider":
            svc.remove_llm_provider("p1")
        elif mutation == "set_roles":
            svc.set_llm_roles({"default": "p1:n", "memory": "p1:n", "council": ["p1:n"]})
        elif mutation == "create_model":
            svc.create_llm_provider_models("p1", {"models": [{"id": "m", "alias": "replacement"}]})
        elif mutation == "update_model":
            svc.update_llm_provider_model("p1", "m", {"alias": "updated"})
        elif mutation == "delete_model":
            svc.delete_llm_provider_model("p1", "m")
        else:  # pragma: no cover - parameter list is exhaustive
            raise AssertionError(mutation)

    with pytest.raises(SecretVaultError, match="vault write failed"):
        mutate()

    assert svc._registry is old_registry
    assert svc._registry.status() == old_status
    assert svc._registry.get("p1").api_key == "sk-old"
    assert svc._llm is old_runtime
    assert svc._memory_settings is old_memory_settings
    assert wire_calls == []
    assert stale_calls == []
