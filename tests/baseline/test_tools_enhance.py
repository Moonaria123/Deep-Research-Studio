"""Prompt enhancement tool: LLM-backed query rewrite via service + HTTP."""

from __future__ import annotations

import pytest

from deepresearch_core.baseline import BaselineService
from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.settings import ProviderSettings

from tests.baseline.fakes import FakeLLM


def _ready_service(reply: str = "Rewritten, sharper research question?") -> BaselineService:
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    # Inject a deterministic LLM and mark the backend ready.
    service._llm = FakeLLM(scripted={"climate": reply})  # noqa: SLF001
    service._orchestrator._llm = service._llm  # noqa: SLF001
    service._llm_ready = True  # noqa: SLF001
    return service


def test_enhance_prompt_returns_rewritten_text():
    service = _ready_service()
    result = service.enhance_prompt({"query": "climate change"})
    assert result["enhanced"] == "Rewritten, sharper research question?"


def test_enhance_prompt_requires_query():
    service = _ready_service()
    with pytest.raises(BaselineError) as exc:
        service.enhance_prompt({"query": "   "})
    assert exc.value.code is ErrorCode.INVALID_REQUEST


def test_enhance_prompt_requires_llm():
    service = BaselineService(provider=ProviderSettings(), db_path=":memory:")
    with pytest.raises(BaselineError) as exc:
        service.enhance_prompt({"query": "anything"})
    assert exc.value.code is ErrorCode.PROVIDER_NOT_CONFIGURED


def test_enhance_prompt_rejects_oversized_query():
    service = _ready_service()
    with pytest.raises(BaselineError) as exc:
        service.enhance_prompt({"query": "x" * 9000})
    assert exc.value.code is ErrorCode.INVALID_REQUEST


# --- HTTP --------------------------------------------------------------------

pytest.importorskip("fastapi.testclient", reason="fastapi not installed")


def test_enhance_endpoint():
    from fastapi.testclient import TestClient

    from deepresearch_core.baseline import create_app

    service = _ready_service()
    client = TestClient(create_app(service))
    res = client.post("/api/v1/tools/enhance", json={"query": "climate change"})
    assert res.status_code == 200
    assert res.json()["enhanced"] == "Rewritten, sharper research question?"
