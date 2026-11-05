"""Exact-origin CORS contract for optional Web development origins."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.settings import ProviderSettings

pytest.importorskip("fastapi.testclient", reason="fastapi not installed")
from fastapi.testclient import TestClient  # noqa: E402


WEB_ORIGIN = "http://localhost:5173"


@pytest.fixture
def service() -> BaselineService:
    return BaselineService(provider=ProviderSettings(), db_path=":memory:")


def _preflight(client: TestClient, origin: str):
    return client.options(
        "/api/v1/sessions",
        headers={
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Authorization, Content-Type",
        },
    )


def test_cors_is_disabled_by_default(service):
    response = _preflight(TestClient(create_app(service)), WEB_ORIGIN)

    assert "access-control-allow-origin" not in response.headers
    assert "access-control-allow-credentials" not in response.headers


def test_cors_allows_only_the_explicit_web_origin_without_credentials(service):
    client = TestClient(create_app(service, cors_origins=(WEB_ORIGIN,)))

    allowed = _preflight(client, WEB_ORIGIN)
    denied = _preflight(client, "http://attacker.invalid")

    assert allowed.status_code == 200
    assert allowed.headers["access-control-allow-origin"] == WEB_ORIGIN
    assert "access-control-allow-credentials" not in allowed.headers
    allow_headers = allowed.headers["access-control-allow-headers"].lower()
    assert "authorization" in allow_headers and "content-type" in allow_headers
    assert "*" not in allow_headers
    assert "access-control-allow-origin" not in denied.headers


@pytest.mark.parametrize(
    "origin",
    [
        "*",
        "http://localhost:5173/path",
        "localhost:5173",
        "http://:",
        "http://127.0.0.1:not-a-port",
        "http://127.0.0.1:70000",
    ],
)
def test_cors_rejects_non_exact_or_unsafe_origins(service, origin):
    with pytest.raises(BaselineError) as exc:
        create_app(service, cors_origins=(origin,))

    assert exc.value.code is ErrorCode.INVALID_REQUEST
