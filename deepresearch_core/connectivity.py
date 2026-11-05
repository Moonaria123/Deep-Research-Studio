import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from .model_listing import list_models
from .settings import ProviderSettings
from .standalone.llm import redact_text


UrlOpener = Callable[[urllib.request.Request, float], Any]


@dataclass(frozen=True)
class ProviderConnectivityResult:
    """Secret-free result for an optional live LLM provider connectivity check."""

    enabled: bool
    ok: bool
    status: str
    latency_ms: int | None = None
    model: str = ""
    endpoint_host: str = ""
    missing_required: list[str] = field(default_factory=list)
    error_message: str = ""
    models_count: int | None = None

    def to_payload(self) -> dict[str, Any]:
        """Return a stable JSON-safe payload without credential values."""

        return {
            "enabled": self.enabled,
            "ok": self.ok,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "model": self.model,
            "endpoint_host": self.endpoint_host,
            "missing_required": list(self.missing_required),
            "error_message": self.error_message,
            "models_count": self.models_count,
        }


@dataclass(frozen=True)
class SearchConnectivityResult:
    """Secret-free result for an optional live search provider connectivity check."""

    enabled: bool
    ok: bool
    status: str
    latency_ms: int | None = None
    quick_answer_ready: bool = False
    report_ready: bool = False
    brave_status: str = "skipped"
    firecrawl_status: str = "skipped"
    brave_host: str = "api.search.brave.com"
    firecrawl_host: str = ""
    missing_required: list[str] = field(default_factory=list)
    error_message: str = ""
    next_step: str = ""

    def to_payload(self) -> dict[str, Any]:
        """Return a stable JSON-safe payload without credential values."""

        return {
            "enabled": self.enabled,
            "ok": self.ok,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "quick_answer_ready": self.quick_answer_ready,
            "report_ready": self.report_ready,
            "brave_status": self.brave_status,
            "firecrawl_status": self.firecrawl_status,
            "brave_host": self.brave_host,
            "firecrawl_host": self.firecrawl_host,
            "missing_required": list(self.missing_required),
            "error_message": self.error_message,
            "next_step": self.next_step,
        }


def format_connectivity_text(result: ProviderConnectivityResult) -> str:
    """Format a connectivity result without exposing credential values."""

    if not result.enabled:
        return "Provider connectivity: skipped"
    if result.ok:
        detail = f"reachable ({result.latency_ms}ms)" if result.latency_ms is not None else "reachable"
        return f"Provider connectivity: {detail}"
    detail = result.status
    if result.error_message:
        detail = f"{detail} - {result.error_message}"
    elif result.missing_required:
        detail = f"{detail} - missing {', '.join(result.missing_required)}"
    return f"Provider connectivity: {detail}"


def format_search_connectivity_text(result: SearchConnectivityResult) -> str:
    """Format a search connectivity result without exposing credential values."""

    if not result.enabled:
        return "Search connectivity: skipped"
    if result.ok:
        detail = f"reachable ({result.latency_ms}ms)" if result.latency_ms is not None else "reachable"
        return (
            f"Search connectivity: {detail}; quick_answer_ready={result.quick_answer_ready}; "
            f"report_ready={result.report_ready}"
        )
    detail = (
        f"{result.status}; brave={result.brave_status}; firecrawl={result.firecrawl_status}; "
        f"quick_answer_ready={result.quick_answer_ready}; report_ready={result.report_ready}"
    )
    if result.error_message:
        detail = f"{detail} - {result.error_message}"
    elif result.missing_required:
        detail = f"{detail} - missing {', '.join(result.missing_required)}"
    if result.next_step:
        detail = f"{detail}; next={result.next_step}"
    return f"Search connectivity: {detail}"


def check_llm_provider_connectivity(
    provider: ProviderSettings,
    *,
    enabled: bool,
    timeout_s: float = 10.0,
    opener: UrlOpener | None = None,
) -> ProviderConnectivityResult:
    """Optionally check an OpenAI-compatible chat endpoint without exposing secrets."""

    if not enabled:
        return ProviderConnectivityResult(enabled=False, ok=False, status="skipped")
    missing = _missing_llm_settings(provider)
    endpoint_host = _endpoint_host(provider.llm_base_url)
    if missing:
        return ProviderConnectivityResult(
            enabled=True,
            ok=False,
            status="missing_config",
            model=_safe_model(provider.llm_model),
            endpoint_host=endpoint_host,
            missing_required=missing,
        )
    url = _chat_completions_url(provider.llm_base_url)
    if not url:
        return ProviderConnectivityResult(
            enabled=True,
            ok=False,
            status="invalid_config",
            model=_safe_model(provider.llm_model),
            endpoint_host=endpoint_host,
            error_message="LLM Base URL must be an http(s) URL",
        )

    request = urllib.request.Request(
        url,
        data=json.dumps(_chat_probe_payload(provider.llm_model), ensure_ascii=False).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {provider.llm_api_key}",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        response = (opener or urllib.request.urlopen)(request, timeout=timeout_s)
        try:
            status_code = int(getattr(response, "status", 0) or response.getcode())
            response.read()
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
    except urllib.error.HTTPError as error:
        return _http_error_result(error, provider, endpoint_host)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return ProviderConnectivityResult(
            enabled=True,
            ok=False,
            status="network_error",
            model=_safe_model(provider.llm_model),
            endpoint_host=endpoint_host,
            error_message=_safe_error_message(error),
        )

    latency_ms = int((time.perf_counter() - started) * 1000)
    if 200 <= status_code < 300:
        return ProviderConnectivityResult(
            enabled=True,
            ok=True,
            status="reachable",
            latency_ms=latency_ms,
            model=_safe_model(provider.llm_model),
            endpoint_host=endpoint_host,
        )
    return ProviderConnectivityResult(
        enabled=True,
        ok=False,
        status="http_error",
        latency_ms=latency_ms,
        model=_safe_model(provider.llm_model),
        endpoint_host=endpoint_host,
        error_message=f"HTTP {status_code}",
    )


def check_provider_models_reachable(
    *,
    type: str,
    base_url: str,
    api_key: str = "",
    timeout_s: float = 5.0,
    opener: UrlOpener | None = None,
) -> ProviderConnectivityResult:
    """Check provider reachability via a model-listing probe (no model id required).

    Unlike :func:`check_llm_provider_connectivity` (a chat-completions probe that
    depends on a specific model id), this verifies ``baseUrl + apiKey + network``
    by listing the catalog (OpenAI-compatible ``GET {base}/models`` or Ollama
    ``GET {base}/api/tags``). It never consumes inference tokens and is the
    web-UI provider-level test. The legacy chat probe is intentionally left
    untouched for the doctor/MCP/CLI/Windows/setup contract paths.
    """

    if not str(base_url or "").strip():
        return ProviderConnectivityResult(
            enabled=True,
            ok=False,
            status="missing_config",
            endpoint_host=_endpoint_host(base_url),
            missing_required=["LLM Base URL"],
        )
    result = list_models(
        type=type or "openai",
        base_url=base_url,
        api_key=api_key,
        timeout_s=timeout_s,
        opener=opener,
    )
    return ProviderConnectivityResult(
        enabled=True,
        ok=result.ok,
        status="reachable" if result.ok else result.status,
        latency_ms=result.latency_ms,
        endpoint_host=result.endpoint_host,
        error_message=result.error_message,
        models_count=len(result.models) if result.ok else None,
    )


def check_search_provider_connectivity(
    provider: ProviderSettings,
    *,
    enabled: bool,
    timeout_s: float = 10.0,
    opener: UrlOpener | None = None,
) -> SearchConnectivityResult:
    """Optionally check Brave search and Firecrawl scrape reachability without exposing secrets."""

    if not enabled:
        return SearchConnectivityResult(enabled=False, ok=False, status="skipped")
    missing = _missing_search_settings(provider)
    firecrawl_host = _endpoint_host(provider.firecrawl_base_url)
    if missing:
        return SearchConnectivityResult(
            enabled=True,
            ok=False,
            status="missing_config",
            firecrawl_host=firecrawl_host,
            missing_required=missing,
            next_step="补齐 Brave 搜索 Key 和 Firecrawl 抓取 Key 后再测试搜索链路。",
        )
    firecrawl_url = _firecrawl_scrape_url(provider.firecrawl_base_url)
    if not firecrawl_url:
        return SearchConnectivityResult(
            enabled=True,
            ok=False,
            status="invalid_config",
            firecrawl_host=firecrawl_host,
            error_message="Firecrawl Base URL must be an http(s) URL",
            next_step="检查 Firecrawl 接口地址，只保留 http(s) scheme 和 host。",
        )

    started = time.perf_counter()
    brave_status, brave_error = _probe_brave_search(provider, timeout_s=timeout_s, opener=opener)
    firecrawl_status, firecrawl_error = _probe_firecrawl_scrape(
        provider,
        firecrawl_url=firecrawl_url,
        timeout_s=timeout_s,
        opener=opener,
    )
    latency_ms = int((time.perf_counter() - started) * 1000)
    quick_ready = brave_status == "reachable"
    report_ready = quick_ready and firecrawl_status == "reachable"
    ok = report_ready
    status = "reachable" if ok else "partial" if quick_ready else "failed"
    error_message = _join_errors(brave_error, firecrawl_error)
    return SearchConnectivityResult(
        enabled=True,
        ok=ok,
        status=status,
        latency_ms=latency_ms,
        quick_answer_ready=quick_ready,
        report_ready=report_ready,
        brave_status=brave_status,
        firecrawl_status=firecrawl_status,
        firecrawl_host=firecrawl_host,
        error_message=error_message,
        next_step=_search_next_step(brave_status, firecrawl_status),
    )


def check_named_search_provider_connectivity(
    provider: ProviderSettings,
    *,
    provider_id: str,
    timeout_s: float = 10.0,
    opener: UrlOpener | None = None,
) -> dict[str, Any]:
    """Probe exactly one search provider and return a secret-free result."""

    selected = str(provider_id or "").strip().lower()
    capabilities = {
        "brave": {"search": True, "extract": False},
        "tavily": {"search": True, "extract": True},
        "exa": {"search": True, "extract": True},
        "firecrawl": {"search": True, "extract": True},
    }
    if selected not in capabilities:
        raise ValueError("provider must be brave, tavily, exa, or firecrawl")
    keys = {
        "brave": provider.brave_api_key,
        "tavily": provider.tavily_api_key,
        "exa": provider.exa_api_key,
        "firecrawl": provider.firecrawl_api_key,
    }
    if not str(keys[selected] or "").strip():
        return {
            "enabled": True,
            "ok": False,
            "status": "missing_config",
            "provider": selected,
            "capabilities": capabilities[selected],
            "latency_ms": None,
            "endpoint_host": _search_provider_host(selected, provider),
            "error_message": "",
            "probe_state": "not_configured",
            "credential_state": "unknown",
            "availability": "unavailable",
            "retry_after": None,
            "reason_codes": ["provider_not_configured"],
        }
    request = _named_search_probe_request(selected, provider)
    started = time.perf_counter()
    fact = _probe_named_search_request(
        request,
        provider_id=selected,
        timeout_s=timeout_s,
        opener=opener,
    )
    return {
        "enabled": True,
        "ok": fact["probe_state"] == "available",
        "status": fact["status"],
        "provider": selected,
        "capabilities": capabilities[selected],
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "endpoint_host": _search_provider_host(selected, provider),
        "error_message": fact["error_message"],
        "probe_state": fact["probe_state"],
        "credential_state": fact["credential_state"],
        "availability": fact["availability"],
        "retry_after": fact.get("retry_after"),
        "reason_codes": list(fact.get("reason_codes") or []),
    }


def _probe_named_search_request(
    request: urllib.request.Request,
    *,
    provider_id: str,
    timeout_s: float,
    opener: UrlOpener | None,
) -> dict[str, Any]:
    """Return a typed, secret-free probe fact for Save/runtime state decisions."""

    try:
        response = (opener or urllib.request.urlopen)(request, timeout=timeout_s)
        try:
            status_code = int(getattr(response, "status", 0) or response.getcode())
            body = response.read(64 * 1024)
            headers = getattr(response, "headers", None)
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
    except urllib.error.HTTPError as error:
        status_code = int(getattr(error, "code", 0) or 0)
        body = _read_http_error_body(error)
        headers = getattr(error, "headers", None) or getattr(error, "hdrs", None)
        return _classify_named_search_response(
            provider_id,
            status_code=status_code,
            body=body,
            headers=headers,
        )
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return {
            "status": "network_error",
            "probe_state": "network_unreachable",
            "credential_state": "unknown",
            "availability": "temporarily_unavailable",
            "retry_after": time.time() + 60.0,
            "reason_codes": ["provider_network_unreachable"],
            "error_message": _safe_error_message(error),
        }
    return _classify_named_search_response(
        provider_id,
        status_code=status_code,
        body=body,
        headers=headers,
    )


def _classify_named_search_response(
    provider_id: str,
    *,
    status_code: int,
    body: bytes,
    headers: Any,
) -> dict[str, Any]:
    parsed = _json_mapping(body)
    authenticated_json = parsed is not None
    retry_after = _retry_after_epoch(headers)
    if 200 <= status_code < 300:
        if _valid_named_search_payload(provider_id, parsed):
            return {
                "status": "reachable",
                "probe_state": "available",
                "credential_state": "accepted",
                "availability": "available",
                "retry_after": None,
                "reason_codes": [],
                "error_message": "",
            }
        return {
            "status": "invalid_response",
            "probe_state": "service_unavailable",
            "credential_state": "unknown",
            "availability": "temporarily_unavailable",
            "retry_after": time.time() + 60.0,
            "reason_codes": ["provider_invalid_success_response"],
            "error_message": "Provider returned an invalid success response.",
        }
    if status_code == 401 or (status_code == 403 and authenticated_json):
        return {
            "status": "auth_failed",
            "probe_state": "credential_rejected",
            "credential_state": "rejected",
            "availability": "unavailable",
            "retry_after": None,
            "reason_codes": ["provider_credential_rejected"],
            "error_message": f"HTTP {status_code}",
        }
    if status_code == 429 and authenticated_json:
        return {
            "status": "rate_limited",
            "probe_state": "rate_limited",
            "credential_state": "accepted",
            "availability": "temporarily_unavailable",
            "retry_after": retry_after or (time.time() + 60.0),
            "reason_codes": ["provider_rate_limited"],
            "error_message": "HTTP 429",
        }
    if status_code == 429:
        return {
            "status": "rate_limited",
            "probe_state": "service_unavailable",
            "credential_state": "unknown",
            "availability": "temporarily_unavailable",
            "retry_after": retry_after or (time.time() + 60.0),
            "reason_codes": ["provider_rate_limit_auth_unknown"],
            "error_message": "HTTP 429",
        }
    if status_code in {402, 432, 433} and authenticated_json:
        return {
            "status": "payment_required",
            "probe_state": "action_required",
            "credential_state": "accepted",
            "availability": "action_required",
            "retry_after": None,
            "reason_codes": ["provider_account_action_required"],
            "error_message": f"HTTP {status_code}",
        }
    if status_code in {402, 432, 433}:
        return {
            "status": "payment_required",
            "probe_state": "service_unavailable",
            "credential_state": "unknown",
            "availability": "temporarily_unavailable",
            "retry_after": retry_after or (time.time() + 60.0),
            "reason_codes": ["provider_account_state_auth_unknown"],
            "error_message": f"HTTP {status_code}",
        }
    if status_code == 403:
        reason = "provider_waf_or_region_block"
    elif 500 <= status_code < 600:
        reason = "provider_service_unavailable"
    else:
        reason = "provider_http_failure"
    return {
        "status": "http_error" if status_code != 403 else "forbidden",
        "probe_state": "service_unavailable",
        "credential_state": "unknown",
        "availability": "temporarily_unavailable",
        "retry_after": retry_after or (time.time() + 60.0),
        "reason_codes": [reason],
        "error_message": f"HTTP {status_code}" if status_code else "Provider request failed.",
    }


def _valid_named_search_payload(provider_id: str, payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    if provider_id == "brave":
        web = payload.get("web")
        return isinstance(web, dict) and isinstance(web.get("results"), list)
    if provider_id in {"tavily", "exa"}:
        return isinstance(payload.get("results"), list)
    return payload.get("success") is True and isinstance(payload.get("data"), list)


def _json_mapping(body: bytes) -> dict[str, Any] | None:
    try:
        parsed = json.loads(body.decode("utf-8", errors="replace"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _read_http_error_body(error: urllib.error.HTTPError) -> bytes:
    try:
        return error.read(64 * 1024)
    except Exception:  # noqa: BLE001 - body is classifier input only
        return b""


def _retry_after_epoch(headers: Any) -> float | None:
    if headers is None:
        return None
    try:
        raw = headers.get("Retry-After")
    except Exception:  # noqa: BLE001 - tolerate test doubles
        raw = None
    try:
        seconds = int(str(raw or "").strip())
    except (TypeError, ValueError):
        return None
    return time.time() + float(max(1, min(3600, seconds)))


def _named_search_probe_request(provider_id: str, provider: ProviderSettings) -> urllib.request.Request:
    query = "DeepResearch connectivity check"
    if provider_id == "brave":
        params = urllib.parse.urlencode({"q": query, "count": 1, "country": "US", "search_lang": "en"})
        return urllib.request.Request(
            f"https://api.search.brave.com/res/v1/web/search?{params}",
            headers={"Accept": "application/json", "X-Subscription-Token": provider.brave_api_key},
            method="GET",
        )
    if provider_id == "tavily":
        url = "https://api.tavily.com/search"
        key = provider.tavily_api_key
        payload = {"query": query, "search_depth": "basic", "max_results": 1}
        headers = {"Authorization": f"Bearer {key}"}
    elif provider_id == "exa":
        url = "https://api.exa.ai/search"
        payload = {"query": query, "numResults": 1, "type": "auto"}
        headers = {"x-api-key": provider.exa_api_key}
    else:
        url = f"{str(provider.firecrawl_base_url or 'https://api.firecrawl.dev').rstrip('/')}/v2/search"
        payload = {"query": query, "limit": 1, "sources": ["web"]}
        headers = {"Authorization": f"Bearer {provider.firecrawl_api_key}"}
    return urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Accept": "application/json", "Content-Type": "application/json", **headers},
        method="POST",
    )


def _search_provider_host(provider_id: str, provider: ProviderSettings) -> str:
    if provider_id == "brave":
        return "api.search.brave.com"
    if provider_id == "tavily":
        return "api.tavily.com"
    if provider_id == "exa":
        return "api.exa.ai"
    return _endpoint_host(provider.firecrawl_base_url)


def _missing_llm_settings(provider: ProviderSettings) -> list[str]:
    """Return only LLM fields needed by the live connectivity probe."""

    missing: list[str] = []
    status = provider.safe_status()
    if not status["llm_base_url"]:
        missing.append("LLM Base URL")
    if not status["llm_api_key"]:
        missing.append("LLM API Key")
    if not status["llm_model"]:
        missing.append("LLM Model")
    return missing


def _missing_search_settings(provider: ProviderSettings) -> list[str]:
    """Return only search fields needed by the live search probe."""

    missing: list[str] = []
    status = provider.safe_status()
    if not status["brave_api_key"]:
        missing.append("Brave API Key")
    if not status["firecrawl_api_key"]:
        missing.append("Firecrawl API Key")
    if not status["firecrawl_base_url"]:
        missing.append("Firecrawl Base URL")
    return missing


def _chat_completions_url(base_url: str) -> str:
    """Return the OpenAI-compatible chat completions URL or an empty string."""

    text = str(base_url or "").strip().rstrip("/")
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        return ""
    return f"{text}/chat/completions"


def _firecrawl_scrape_url(base_url: str) -> str:
    """Return the Firecrawl scrape URL or an empty string."""

    text = str(base_url or "").strip().rstrip("/")
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        return ""
    return f"{text}/v2/scrape"


def _chat_probe_payload(model: str) -> dict[str, Any]:
    """Return a minimal provider probe request body."""

    return {
        "model": model.strip(),
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 16,
        "temperature": 0,
        "stream": False,
    }


def _probe_brave_search(provider: ProviderSettings, *, timeout_s: float, opener: UrlOpener | None) -> tuple[str, str]:
    query = urllib.parse.urlencode({"q": "DeepResearch connectivity check", "count": 1, "country": "US", "search_lang": "en"})
    request = urllib.request.Request(
        f"https://api.search.brave.com/res/v1/web/search?{query}",
        headers={"Accept": "application/json", "X-Subscription-Token": provider.brave_api_key},
        method="GET",
    )
    return _probe_request_status(request, timeout_s=timeout_s, opener=opener)


def _probe_firecrawl_scrape(
    provider: ProviderSettings,
    *,
    firecrawl_url: str,
    timeout_s: float,
    opener: UrlOpener | None,
) -> tuple[str, str]:
    request = urllib.request.Request(
        firecrawl_url,
        data=json.dumps({"url": "https://example.com", "formats": ["markdown"], "onlyMainContent": True}, ensure_ascii=False).encode(
            "utf-8"
        ),
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Authorization": f"Bearer {provider.firecrawl_api_key}",
        },
        method="POST",
    )
    return _probe_request_status(request, timeout_s=timeout_s, opener=opener)


def _probe_request_status(request: urllib.request.Request, *, timeout_s: float, opener: UrlOpener | None) -> tuple[str, str]:
    try:
        response = (opener or urllib.request.urlopen)(request, timeout=timeout_s)
        try:
            status_code = int(getattr(response, "status", 0) or response.getcode())
            response.read()
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
    except urllib.error.HTTPError as error:
        status_code = int(getattr(error, "code", 0) or 0)
        status = _search_http_failure_status(status_code)
        return status, f"HTTP {status_code}" if status_code else _safe_error_message(error)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return "network_error", _safe_error_message(error)
    if 200 <= status_code < 300:
        return "reachable", ""
    return "http_error", f"HTTP {status_code}"


def _search_http_failure_status(status_code: int) -> str:
    if status_code == 401:
        return "auth_failed"
    if status_code in {402, 432, 433}:
        return "payment_required"
    if status_code == 403:
        return "forbidden"
    if status_code == 429:
        return "rate_limited"
    return "http_error"


def _http_error_result(
    error: urllib.error.HTTPError,
    provider: ProviderSettings,
    endpoint_host: str,
) -> ProviderConnectivityResult:
    """Return a redacted structured HTTP failure."""

    status_code = int(getattr(error, "code", 0) or 0)
    status = "auth_failed" if status_code in {401, 403} else "http_error"
    return ProviderConnectivityResult(
        enabled=True,
        ok=False,
        status=status,
        model=_safe_model(provider.llm_model),
        endpoint_host=endpoint_host,
        error_message=f"HTTP {status_code}" if status_code else _safe_error_message(error),
    )


def _join_errors(*messages: str) -> str:
    errors = [message for message in (_safe_error_text(item) for item in messages) if message]
    return "; ".join(errors)[:240]


def _safe_error_text(value: str) -> str:
    return redact_text(str(value or "").strip())[:120]


def _search_next_step(brave_status: str, firecrawl_status: str) -> str:
    if brave_status != "reachable":
        return "先检查 Brave 搜索 Key、额度和网络；普通问答需要 Brave 搜索可用。"
    if firecrawl_status != "reachable":
        return "Brave 可用但 Firecrawl 不可用；普通问答可先运行，深度报告需修复 Firecrawl 抓取。"
    return ""


def _endpoint_host(base_url: str) -> str:
    """Return the endpoint host only, without path, query, or credentials."""

    parsed = urlparse(str(base_url or "").strip())
    return parsed.hostname or ""


def _safe_model(model: str) -> str:
    """Return a model label safe for display."""

    return redact_text(str(model or "").strip())


def _safe_error_message(error: BaseException) -> str:
    """Return a compact redacted error message."""

    text = redact_text(str(error))
    if len(text) > 240:
        return f"{text[:237]}..."
    return text
