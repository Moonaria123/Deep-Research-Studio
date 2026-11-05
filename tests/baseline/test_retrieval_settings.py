"""DR-ITER-089: retrieval settings API and config persistence."""

from __future__ import annotations

import json

import pytest

from deepresearch_core.baseline import create_app
from deepresearch_core.baseline.service import BaselineService
from deepresearch_core.settings import ProviderSettings


pytest.importorskip("fastapi.testclient", reason="fastapi not installed")
from fastapi.testclient import TestClient  # noqa: E402


def test_retrieval_settings_default_to_chroma_persistence(tmp_path):
    service = BaselineService(
        provider=ProviderSettings(),
        db_path=tmp_path / "deepresearch.sqlite",
        config_path=tmp_path / "config.json",
    )

    settings = service.get_retrieval_settings()

    assert settings["saved"]["chroma_persist_enabled"] is True
    assert settings["saved"]["kg_aware_retrieval_enabled"] is False
    assert settings["effective"]["chroma_persist_enabled"] is True
    assert settings["effective"]["kg_aware_retrieval_enabled"] is False
    assert settings["effective"]["chroma_persist_mode"] == "disk"
    assert settings["effective"]["chroma_persist_path"].endswith("/chroma")
    service.store.close()


def test_retrieval_settings_patch_preserves_other_config_blocks(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"providers": {"sentinel": True}}), encoding="utf-8")
    service = BaselineService(
        provider=ProviderSettings(),
        db_path=tmp_path / "deepresearch.sqlite",
        config_path=config_path,
    )
    client = TestClient(create_app(service, auth_mode="off"))

    response = client.patch("/api/v1/settings/retrieval", json={"chroma_persist_enabled": False})

    assert response.status_code == 200
    body = response.json()
    assert body["saved"]["chroma_persist_enabled"] is False
    assert body["effective"]["chroma_persist_enabled"] is False
    assert body["restart_required"] is True
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["providers"] == {"sentinel": True}
    assert persisted["retrieval"] == {
        "chroma_persist_enabled": False,
        "kg_aware_retrieval_enabled": False,
    }
    service.store.close()


def test_kg_retrieval_toggle_is_immediate_and_preserves_future_keys(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "retrieval": {
                    "chroma_persist_enabled": True,
                    "future_retrieval_policy": {"limit": 7},
                }
            }
        ),
        encoding="utf-8",
    )
    service = BaselineService(
        provider=ProviderSettings(),
        db_path=tmp_path / "deepresearch.sqlite",
        config_path=config_path,
    )
    client = TestClient(create_app(service, auth_mode="off"))

    response = client.patch(
        "/api/v1/settings/retrieval",
        json={"kg_aware_retrieval_enabled": True},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["saved"]["kg_aware_retrieval_enabled"] is True
    assert body["effective"]["kg_aware_retrieval_enabled"] is True
    assert body["restart_required"] is False
    persisted = json.loads(config_path.read_text(encoding="utf-8"))
    assert persisted["retrieval"]["future_retrieval_policy"] == {"limit": 7}
    service.store.close()


def test_retrieval_settings_health_uses_secret_free_path(tmp_path):
    service = BaselineService(
        provider=ProviderSettings(),
        db_path=tmp_path / "deepresearch.sqlite",
        config_path=tmp_path / "config.json",
    )
    client = TestClient(create_app(service, auth_mode="off"))

    health = client.get("/api/v1/health").json()

    assert "chroma_persist_enabled" in health
    assert "chroma_persist_mode" in health
    assert str(tmp_path) not in str(health.get("chroma_persist_path"))
    service.store.close()


def test_retrieval_settings_reject_invalid_payload(tmp_path):
    service = BaselineService(
        provider=ProviderSettings(),
        db_path=tmp_path / "deepresearch.sqlite",
        config_path=tmp_path / "config.json",
    )
    client = TestClient(create_app(service, auth_mode="off"))

    response = client.patch("/api/v1/settings/retrieval", json={"chroma_persist_enabled": "yes"})

    assert response.status_code == 400
    service.store.close()
