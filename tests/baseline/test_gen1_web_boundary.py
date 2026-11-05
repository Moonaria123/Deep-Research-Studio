"""Browser-origin boundaries for the loopback-only web distribution."""

import pytest
from fastapi.testclient import TestClient

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.settings import ProviderSettings


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("DEEPRESEARCH_LOCAL_WEB_GUARD", "1")
    monkeypatch.setenv("DEEPRESEARCH_WEB_ORIGINS", "http://127.0.0.1:15174,http://localhost:15174")
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    with TestClient(create_app(service), base_url="http://127.0.0.1:18788") as api:
        yield api


def test_local_browser_and_health_probe_allowed(client):
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/providers/llm", headers={"Origin": "http://localhost:15174"}).status_code == 200


@pytest.mark.parametrize("headers", [
    {"Host": "evil.example"},
    {"Host": "localhost.evil.example"},
    {"Host": "user@localhost"},
    {"Host": "localhost:invalid"},
    {"Origin": "http://evil.example"},
    {"Origin": "null"},
    {"Origin": "http://127.0.0.1:9999"},
    {"Sec-Fetch-Site": "cross-site"},
])
def test_rebinding_and_cross_site_read_write_rejected(client, headers):
    assert client.get("/api/v1/providers/llm", headers=headers).status_code == 403
    assert client.put("/api/v1/providers/llm", headers=headers, json={"id": "injected"}).status_code == 403
    assert not client.get("/api/v1/providers/llm").json()["providers"]


def test_vault_error_is_safe_and_actionable(client, monkeypatch):
    from deepresearch_core.secret_vault import SecretVaultError

    def locked(*_args, **_kwargs):
        raise SecretVaultError("private diagnostic must not reach HTTP")

    monkeypatch.setattr(BaselineService, "upsert_llm_provider", locked)
    response = client.put("/api/v1/providers/llm", json={"id": "test"})
    assert response.status_code == 503
    assert "private diagnostic" not in response.text
