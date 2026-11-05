"""API bearer-token authentication (DR-ITER-027)."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.settings import ProviderSettings

pytest.importorskip("fastapi.testclient", reason="fastapi not installed")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def service() -> BaselineService:
    return BaselineService(provider=ProviderSettings(), db_path=":memory:")


def test_off_mode_allows_unauthenticated_requests(service):
    # Default mode mirrors today's behavior: no Authorization header required.
    client = TestClient(create_app(service, auth_mode="off"))
    assert client.get("/api/v1/health").status_code == 200
    assert client.get("/api/v1/sessions").status_code == 200


def test_token_mode_rejects_missing_token(service):
    client = TestClient(create_app(service, auth_mode="token", auth_token="s3cret"))
    res = client.get("/api/v1/sessions")
    assert res.status_code == 401
    assert res.json()["error"]["code"] == ErrorCode.UNAUTHORIZED.value


def test_token_mode_rejects_wrong_token(service):
    client = TestClient(create_app(service, auth_mode="token", auth_token="s3cret"))
    res = client.get("/api/v1/sessions", headers={"Authorization": "Bearer nope"})
    assert res.status_code == 401


def test_token_mode_accepts_valid_token(service):
    client = TestClient(create_app(service, auth_mode="token", auth_token="s3cret"))
    res = client.get("/api/v1/sessions", headers={"Authorization": "Bearer s3cret"})
    assert res.status_code == 200


def test_health_stays_public_in_token_mode(service):
    # Health must stay open so Docker/sidecar probes work without credentials.
    client = TestClient(create_app(service, auth_mode="token", auth_token="s3cret"))
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.json()["auth_mode"] == "token"


def test_token_mode_without_token_fails_fast(service):
    # Refuse to start an "authenticated" server that would accept everything.
    with pytest.raises(BaselineError) as exc:
        create_app(service, auth_mode="token", auth_token="")
    assert exc.value.code is ErrorCode.INVALID_REQUEST


def test_invalid_mode_fails_fast(service):
    with pytest.raises(BaselineError) as exc:
        create_app(service, auth_mode="bogus")
    assert exc.value.code is ErrorCode.INVALID_REQUEST
