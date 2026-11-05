"""Model auto-fetch: list available models from an LLM provider, secret-free.

Supports two transports:
- OpenAI-compatible: ``GET {base_url}/models`` (Bearer auth) -> ``data[].id``
- Ollama: ``GET {base_url}/api/tags`` -> ``models[].name``

All results are credential-free; raw keys never appear in payloads or errors.
The ``opener`` seam mirrors :mod:`deepresearch_core.connectivity` so tests stay
hermetic (no live network).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from .standalone.llm import redact_text

UrlOpener = Callable[[urllib.request.Request, float], Any]


@dataclass(frozen=True)
class ModelListResult:
    """Secret-free result of a model-listing probe."""

    ok: bool
    status: str
    models: list[str] = field(default_factory=list)
    latency_ms: int | None = None
    endpoint_host: str = ""
    error_message: str = ""

    def to_payload(self) -> dict[str, Any]:
        """Return a stable JSON-safe payload without credential values."""

        return {
            "ok": self.ok,
            "status": self.status,
            "models": list(self.models),
            "latency_ms": self.latency_ms,
            "endpoint_host": self.endpoint_host,
            "error_message": self.error_message,
        }


def list_models(
    *,
    type: str,
    base_url: str,
    api_key: str = "",
    timeout_s: float = 10.0,
    opener: UrlOpener | None = None,
) -> ModelListResult:
    """List model ids for an LLM provider candidate, without exposing secrets."""

    provider_type = str(type or "openai").strip().lower() or "openai"
    base = str(base_url or "").strip().rstrip("/")
    host = _host(base)
    if not base:
        return ModelListResult(ok=False, status="missing_config", error_message="Base URL is required")

    if provider_type == "ollama":
        url = f"{base}/api/tags"
        request = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
        extract = _extract_ollama
    else:
        if not _is_http_url(base):
            return ModelListResult(ok=False, status="invalid_config", endpoint_host=host,
                                   error_message="Base URL must be an http(s) URL")
        url = f"{base}/models"
        headers = {"Accept": "application/json"}
        if str(api_key or "").strip():
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(url, headers=headers, method="GET")
        extract = _extract_openai

    started = time.perf_counter()
    try:
        response = (opener or urllib.request.urlopen)(request, timeout=timeout_s)
        try:
            status_code = int(getattr(response, "status", 0) or response.getcode())
            raw = response.read()
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
    except urllib.error.HTTPError as error:
        code = int(getattr(error, "code", 0) or 0)
        status = "auth_failed" if code in {401, 403} else "http_error"
        return ModelListResult(ok=False, status=status, endpoint_host=host,
                               error_message=f"HTTP {code}" if code else _safe_error(error))
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return ModelListResult(ok=False, status="network_error", endpoint_host=host,
                               error_message=_safe_error(error))

    latency_ms = int((time.perf_counter() - started) * 1000)
    if not (200 <= status_code < 300):
        return ModelListResult(ok=False, status="http_error", latency_ms=latency_ms,
                               endpoint_host=host, error_message=f"HTTP {status_code}")
    try:
        data = json.loads(raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return ModelListResult(ok=False, status="bad_response", latency_ms=latency_ms,
                               endpoint_host=host, error_message="Malformed model list response")
    models = extract(data)
    return ModelListResult(ok=True, status="ok", models=models, latency_ms=latency_ms, endpoint_host=host)


def _extract_openai(data: Any) -> list[str]:
    items = data.get("data") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        mid = str((item or {}).get("id") or "").strip() if isinstance(item, dict) else ""
        if mid and mid not in out:
            out.append(mid)
    return out


def _extract_ollama(data: Any) -> list[str]:
    items = data.get("models") if isinstance(data, dict) else None
    if not isinstance(items, list):
        return []
    out: list[str] = []
    for item in items:
        name = str((item or {}).get("name") or "").strip() if isinstance(item, dict) else ""
        if name and name not in out:
            out.append(name)
    return out


def _is_http_url(text: str) -> bool:
    parsed = urlparse(text)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc) and not parsed.username and not parsed.password


def _host(base_url: str) -> str:
    return urlparse(str(base_url or "").strip()).hostname or ""


def _safe_error(error: BaseException) -> str:
    text = redact_text(str(error))
    return f"{text[:237]}..." if len(text) > 240 else text
