"""Phase conn-test: service.test_llm_provider + test_search_tools (secret-free)."""

from __future__ import annotations

import json
import urllib.error
from io import BytesIO
from pathlib import Path

from deepresearch_core.baseline.service import BaselineService
from deepresearch_core.settings import ProviderSettings


class _Resp:
    def __init__(self, status=200, body=b"{}"):
        self.status = status
        self._body = BytesIO(body)

    def read(self):
        return self._body.read()

    def close(self):
        pass


def _service(tmp_path: Path, providers=None, roles=None, provider=None) -> BaselineService:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"llm_providers": providers or [], "llm_roles": roles or {}}), encoding="utf-8")
    return BaselineService(provider=provider or ProviderSettings(), db_path=":memory:", config_path=cfg)


def test_llm_provider_default_probe_lists_models(tmp_path):
    """Provider-level test needs no model id; it lists the catalog via GET models."""

    svc = _service(tmp_path)
    seen: list[tuple[str, str]] = []

    def opener(request, timeout):
        seen.append((request.full_url, request.get_method()))
        return _Resp(body=json.dumps({"data": [{"id": "m1"}, {"id": "m2"}]}).encode())

    out = svc.test_llm_provider(
        {"type": "openai", "base_url": "https://llm.example/v1", "api_key": "sk-secret"}, opener=opener
    )
    assert out["ok"] and out["status"] == "reachable"
    assert out["models_count"] == 2
    assert out["model"] == ""
    assert seen == [("https://llm.example/v1/models", "GET")]
    assert "sk-secret" not in json.dumps(out)


def test_llm_provider_default_probe_auth_failure_is_redacted(tmp_path):
    svc = _service(tmp_path)

    def opener(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized sk-secret", hdrs=None, fp=None)

    out = svc.test_llm_provider(
        {"type": "openai", "base_url": "https://llm.example/v1", "api_key": "sk-secret"}, opener=opener
    )
    assert not out["ok"] and out["status"] == "auth_failed"
    assert "sk-secret" not in json.dumps(out)


def test_llm_provider_models_probe_keeps_base_path(tmp_path):
    """Volcengine regression: {base}/models is concatenated directly (no /v1 inserted)."""

    svc = _service(tmp_path)
    seen: list[str] = []

    def opener(request, timeout):
        seen.append(request.full_url)
        return _Resp(body=json.dumps({"data": [{"id": "deepseek-v3"}]}).encode())

    svc.test_llm_provider(
        {"type": "openai", "base_url": "https://ark.cn-beijing.volces.com/api/v3", "api_key": "sk"},
        opener=opener,
    )
    assert seen == ["https://ark.cn-beijing.volces.com/api/v3/models"]


def test_llm_provider_chat_probe_posts_chat_completions(tmp_path):
    """Model-level probe (probe=chat) requires a model and POSTs chat/completions."""

    svc = _service(tmp_path)
    seen: list[tuple[str, str]] = []

    def opener(request, timeout):
        seen.append((request.full_url, request.get_method()))
        return _Resp(body=b'{"choices":[{"message":{"content":"OK"}}]}')

    out = svc.test_llm_provider(
        {
            "type": "openai",
            "base_url": "https://llm.example/v1",
            "api_key": "sk-secret",
            "probe": "chat",
            "model": "m1",
        },
        opener=opener,
    )
    assert out["ok"] and out["status"] == "reachable" and out["model"] == "m1"
    assert seen == [("https://llm.example/v1/chat/completions", "POST")]
    assert "sk-secret" not in json.dumps(out)


def test_llm_provider_chat_probe_distinguishes_404_from_auth(tmp_path):
    """A model-not-found 404 stays http_error; 401/403 maps to auth_failed."""

    svc = _service(tmp_path)

    def opener_404(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", hdrs=None, fp=None)

    out404 = svc.test_llm_provider(
        {"base_url": "https://llm.example/v1", "api_key": "sk", "probe": "chat", "model": "ghost"},
        opener=opener_404,
    )
    assert not out404["ok"] and out404["status"] == "http_error"

    def opener_401(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", hdrs=None, fp=None)

    out401 = svc.test_llm_provider(
        {"base_url": "https://llm.example/v1", "api_key": "sk", "probe": "chat", "model": "m"},
        opener=opener_401,
    )
    assert not out401["ok"] and out401["status"] == "auth_failed"


def test_llm_provider_ollama_uses_tags_probe(tmp_path):
    svc = _service(tmp_path)
    seen = []

    def opener(request, timeout):
        seen.append(request.full_url)
        return _Resp(body=json.dumps({"models": [{"name": "llama3.1"}]}).encode())

    out = svc.test_llm_provider({"type": "ollama", "base_url": "http://127.0.0.1:11434"}, opener=opener)
    assert out["ok"] and out["status"] == "reachable"
    assert seen == ["http://127.0.0.1:11434/api/tags"]


def test_search_tools_reachable_with_candidate(tmp_path):
    svc = _service(tmp_path)

    def opener(request, timeout):
        return _Resp(body=b"{}")

    out = svc.test_search_tools(
        {"brave_api_key": "bk", "firecrawl_api_key": "fk", "firecrawl_base_url": "https://api.firecrawl.dev"},
        opener=opener,
    )
    assert out["ok"] and out["status"] == "reachable"
    assert out["quick_answer_ready"] and out["report_ready"]
    assert "bk" not in json.dumps(out) and "fk" not in json.dumps(out)


def test_search_tools_missing_config_reports_secret_free(tmp_path):
    svc = _service(tmp_path)
    out = svc.test_search_tools({}, opener=lambda r, t: _Resp())
    assert not out["ok"]
    assert out["status"] in {"missing_config", "failed", "partial"}


def test_saved_key_not_sent_to_caller_overridden_base_url(tmp_path):
    """A saved profile's key must never be probed against a caller-supplied host."""

    providers = [
        {
            "id": "openai",
            "label": "OpenAI",
            "type": "openai",
            "base_url": "https://api.openai.com/v1",
            "api_key": "sk-saved-secret",
            "models": ["gpt-4o-mini"],
            "enabled": True,
        }
    ]
    svc = _service(tmp_path, providers=providers, roles={"default": "openai:gpt-4o-mini"})
    seen: list[tuple[str, str | None]] = []

    def opener(request, timeout):
        seen.append((request.full_url, request.get_header("Authorization")))
        return _Resp(body=b'{"choices":[{"message":{"content":"OK"}}]}')

    # Caller tries to reuse the saved profile but redirect it to a foreign host.
    svc.test_llm_provider(
        {"provider_id": "openai", "base_url": "https://attacker.example/v1"}, opener=opener
    )

    assert seen, "probe should have been attempted"
    url, _auth = seen[0]
    # The saved key may only be sent to the saved host, never the override.
    assert "api.openai.com" in url
    assert "attacker.example" not in url
