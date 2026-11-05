"""Pre-run cost estimate contract tests."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.baseline.cost_estimate import (
    RunCostEstimateRequest,
    estimate_run_cost,
)
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.settings import ProviderSettings


def test_estimate_request_does_not_require_query() -> None:
    request = RunCostEstimateRequest.from_payload({"mode": "deep", "budget": {"max_searches": 2}})

    assert request.mode.value == "deep"
    assert request.resolved_budget().max_searches == 2


def test_static_openai_and_brave_estimate_is_secret_free(monkeypatch) -> None:
    monkeypatch.setattr("deepresearch_core.baseline.cost_estimate._estimate_with_litellm", lambda *_, **__: None)

    request = RunCostEstimateRequest.from_payload({"mode": "deep"})
    estimate = estimate_run_cost(request, model_id="gpt-4o-mini", provider_type="openai", search_provider="brave")
    body = estimate.to_dict()

    assert body["searches"] == 6
    assert body["recovery_searches_max"] == 1
    assert body["search_usd_low"] == pytest.approx(0.035)
    assert body["llm_input_tokens_est"] == 16000
    assert body["llm_output_tokens_est"] == 7200
    assert body["llm_usd_low"] == pytest.approx(0.00672)
    assert body["total_usd_low"] == pytest.approx(0.04172)
    assert "api_key" not in str(body).lower()


def test_firecrawl_estimate_reports_credits_not_usd(monkeypatch) -> None:
    monkeypatch.setattr("deepresearch_core.baseline.cost_estimate._estimate_with_litellm", lambda *_, **__: None)

    request = RunCostEstimateRequest.from_payload({"mode": "multiple"})
    body = estimate_run_cost(
        request,
        model_id="unknown-model",
        provider_type="openai",
        search_provider="firecrawl",
    ).to_dict()

    assert body["search_usd_low"] is None
    assert body["search_credits_low"] == 10.0
    assert body["total_usd_low"] is None
    assert any("credits" in item.lower() for item in body["disclaimers"])


def test_council_strategy_scales_token_estimate() -> None:
    request = RunCostEstimateRequest.from_payload(
        {"mode": "council", "council_size": 3, "council_strategy": "debate_once"}
    )
    body = estimate_run_cost(request, model_id="", provider_type="", search_provider="none").to_dict()

    assert body["llm_input_tokens_est"] == 28_000
    assert body["llm_output_tokens_est"] == 9_800
    assert body["recovery_searches_max"] == 0


def test_invalid_estimate_payload_raises_invalid_request() -> None:
    with pytest.raises(BaselineError) as exc:
        RunCostEstimateRequest.from_payload({"mode": "turbo"})

    assert exc.value.code is ErrorCode.INVALID_REQUEST


def test_service_estimate_cost_returns_unknown_without_provider() -> None:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")

    body = service.estimate_run_cost({"mode": "fast"})

    assert body["mode"] == "fast"
    assert body["confidence"] == "unknown"
    assert body["llm_usd_low"] is None
    assert body["search_usd_low"] is None


fastapi_testclient = pytest.importorskip("fastapi.testclient", reason="fastapi not installed")


def test_estimate_cost_http_route_is_available_without_query(monkeypatch) -> None:
    monkeypatch.setattr("deepresearch_core.baseline.cost_estimate._estimate_with_litellm", lambda *_, **__: None)
    service = BaselineService(
        provider=ProviderSettings(
            llm_base_url="https://api.openai.com/v1",
            llm_api_key="test-key",
            llm_model="gpt-4o-mini",
            brave_api_key="brave-test",
        ),
        db_path=":memory:",
    )
    from fastapi.testclient import TestClient

    client = TestClient(create_app(service))
    res = client.post("/api/v1/runs/estimate-cost", json={"mode": "fast"})

    assert res.status_code == 200
    body = res.json()
    assert body["search_usd_low"] == pytest.approx(0.01)
    assert body["llm_usd_low"] is not None
