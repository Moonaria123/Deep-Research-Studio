"""Phase 3: model auto-fetch (OpenAI /models + Ollama /api/tags), secret-free."""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO

from deepresearch_core.model_listing import list_models


class _Resp:
    def __init__(self, status=200, body=b"{}"):
        self.status = status
        self._body = BytesIO(body)

    def read(self):
        return self._body.read()

    def close(self):
        pass


def test_openai_lists_models_and_sends_bearer():
    calls = []

    def opener(request, timeout):
        calls.append(request)
        return _Resp(body=json.dumps({"data": [{"id": "gpt-4o"}, {"id": "gpt-4o-mini"}]}).encode())

    result = list_models(type="openai", base_url="https://api.openai.com/v1", api_key="sk-secret", opener=opener)

    assert result.ok and result.status == "ok"
    assert result.models == ["gpt-4o", "gpt-4o-mini"]
    assert calls[0].full_url == "https://api.openai.com/v1/models"
    assert calls[0].headers.get("Authorization") == "Bearer sk-secret"
    # Secret never leaks into the payload.
    assert "sk-secret" not in json.dumps(result.to_payload())


def test_ollama_lists_tags_without_auth():
    def opener(request, timeout):
        assert request.full_url == "http://127.0.0.1:11434/api/tags"
        assert "Authorization" not in request.headers
        return _Resp(body=json.dumps({"models": [{"name": "llama3.1:8b"}, {"name": "qwen2"}]}).encode())

    result = list_models(type="ollama", base_url="http://127.0.0.1:11434", opener=opener)
    assert result.ok and result.models == ["llama3.1:8b", "qwen2"]


def test_auth_failure_is_structured_and_redacted():
    def opener(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized sk-secret", hdrs=None, fp=None)

    result = list_models(type="openai", base_url="https://api.openai.com/v1", api_key="sk-secret", opener=opener)
    assert not result.ok and result.status == "auth_failed"
    assert result.models == []
    assert "sk-secret" not in json.dumps(result.to_payload())


def test_network_error_is_normalized():
    def opener(request, timeout):
        raise urllib.error.URLError("connection refused")

    result = list_models(type="openai", base_url="https://nope.example/v1", api_key="k", opener=opener)
    assert not result.ok and result.status == "network_error"


def test_malformed_response_is_handled():
    def opener(request, timeout):
        return _Resp(body=b"not-json")

    result = list_models(type="openai", base_url="https://api.openai.com/v1", api_key="k", opener=opener)
    assert not result.ok and result.status == "bad_response"


def test_invalid_base_url_rejected_without_network():
    calls = []
    result = list_models(type="openai", base_url="ftp://bad", api_key="k",
                         opener=lambda r, t: calls.append(r))
    assert not result.ok and result.status == "invalid_config"
    assert calls == []
