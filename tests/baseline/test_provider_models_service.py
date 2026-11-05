"""Phase 3: service.list_provider_models (candidate + saved-profile, secret-free)."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

import pytest

from deepresearch_core.baseline.errors import BaselineError, ErrorCode
from deepresearch_core.baseline.service import BaselineService
from deepresearch_core.settings import ProviderSettings


class _Resp:
    def __init__(self, body: bytes):
        self.status = 200
        self._body = BytesIO(body)

    def read(self):
        return self._body.read()

    def close(self):
        pass


def _service(tmp_path: Path, providers: list[dict], roles: dict) -> BaselineService:
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps({"llm_providers": providers, "llm_roles": roles}), encoding="utf-8")
    return BaselineService(provider=ProviderSettings(), db_path=":memory:", config_path=cfg)


def test_candidate_fetch_sends_bearer_and_returns_models(tmp_path):
    svc = _service(tmp_path, providers=[], roles={})
    seen = []

    def opener(request, timeout):
        seen.append(request)
        return _Resp(json.dumps({"data": [{"id": "m1"}, {"id": "m2"}]}).encode())

    out = svc.list_provider_models(
        {"type": "openai", "base_url": "https://x.example/v1", "api_key": "sk-candidate"}, opener=opener
    )
    assert out["ok"] and out["models"] == ["m1", "m2"]
    assert seen[0].headers.get("Authorization") == "Bearer sk-candidate"
    assert "sk-candidate" not in json.dumps(out)


def test_saved_profile_reuses_stored_key(tmp_path):
    svc = _service(
        tmp_path,
        providers=[{"id": "p1", "type": "openai", "base_url": "https://p1.example/v1",
                    "api_key": "sk-stored", "models": ["old"], "enabled": True}],
        roles={"default": "p1:old", "embed": "", "council": []},
    )
    seen = []

    def opener(request, timeout):
        seen.append(request)
        return _Resp(json.dumps({"data": [{"id": "new-1"}]}).encode())

    out = svc.list_provider_models({"provider_id": "p1"}, opener=opener)
    assert out["ok"] and out["models"] == ["new-1"]
    assert seen[0].full_url == "https://p1.example/v1/models"
    assert seen[0].headers.get("Authorization") == "Bearer sk-stored"


def test_missing_base_url_raises_invalid_request(tmp_path):
    svc = _service(tmp_path, providers=[], roles={})
    with pytest.raises(BaselineError) as exc:
        svc.list_provider_models({"type": "openai"}, opener=lambda r, t: None)
    assert exc.value.code is ErrorCode.INVALID_REQUEST


def test_unknown_provider_id_raises(tmp_path):
    svc = _service(tmp_path, providers=[], roles={})
    with pytest.raises(BaselineError) as exc:
        svc.list_provider_models({"provider_id": "nope"}, opener=lambda r, t: None)
    assert exc.value.code is ErrorCode.INVALID_REQUEST
