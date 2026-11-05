import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


JsonDict = Dict[str, Any]
Transport = Callable[[urllib.request.Request], JsonDict]
Sleeper = Callable[[float], None]

_RETRY_HTTP_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
_NEVER_RETRY_HTTP_STATUSES = frozenset({401, 403})
_RETRY_TEXT_MARKERS = (
    "301001",
    "rate limit",
    "ratelimit",
    "too many requests",
    "temporarily unavailable",
    "temporary unavailable",
    "try again later",
    "timeout",
    "timed out",
    "connection reset",
    "资源限制",
    "临时不可用",
    "请求过多",
    "稍后重试",
)
_PRIVATE_KEY_BLOCK_RE = re.compile(
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
    re.DOTALL | re.IGNORECASE,
)
_SENSITIVE_CONTENT_MARKERS = (
    "sensitivecontentdetected",
    "may contain sensitive information",
    "contains sensitive information",
)
_TIMEOUT_MARKERS = (
    "read operation timed out",
    "operation timed out",
    "timed out",
    "timeout",
    "响应超时",
    "读取超时",
    "未在时限内",
    "超时",
)
SENSITIVE_CONTENT_USER_MESSAGE = (
    "模型服务拒绝了本次请求：上游安全策略认为输入可能包含敏感信息。"
    "DeepResearch 没有保存这段请求内容。请删去密钥、账号、证件号、手机号、内部链接或其他私密片段后重试；"
    "如果这是误判，可以换一种更概括的表述。"
)
PROVIDER_TIMEOUT_USER_MESSAGE = (
    "上游模型或搜索服务响应超时，本次没有完成。请稍后重试，或在设置中切换更稳定的模型/搜索服务。"
)


@dataclass(frozen=True)
class LLMChatResult:
    """Structured Chat Completions result retained for provider diagnostics."""

    text: str
    model: str
    finish_reason: str | None = None
    usage: JsonDict = field(default_factory=dict)
    malformed_terminal: bool = False


def _current_provider_budget():
    """Lazily read the baseline deadline context without creating an import cycle."""

    from ..baseline.deadline import current_provider_call_budget

    return current_provider_call_budget()


def _provider_budget_context(budget):
    """Lazily install an attempt budget without importing baseline at module load."""

    from ..baseline.deadline import use_provider_call_budget

    return use_provider_call_budget(budget)


class MissingCredentialError(ValueError):
    """Raised when BYOK LLM credentials are missing."""


class ProviderRequestError(RuntimeError):
    """Raised when a provider request fails."""

    def __init__(self, message: Any, *, status_code: Optional[int] = None, retryable: bool = False) -> None:
        super().__init__(redact_text(message))
        self.status_code = status_code
        self.retryable = retryable


@dataclass(frozen=True)
class RetryPolicy:
    """Provider retry policy for transient HTTP and network failures."""

    max_retries: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 4.0
    retry_http_statuses: frozenset[int] = field(default_factory=lambda: _RETRY_HTTP_STATUSES)

    def delay_for(self, retry_index: int) -> float:
        """Return the delay before the next retry attempt."""

        safe_index = max(0, retry_index)
        return min(max(0.0, self.max_delay_s), max(0.0, self.base_delay_s) * (2**safe_index))


def join_url(base_url: str, path: str) -> str:
    """Join a base URL and path without double slashes."""

    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def validate_http_url(value: str, *, field_name: str) -> str:
    """Validate an HTTP(S) URL."""

    parsed = urllib.parse.urlparse((value or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"{field_name} must be an http(s) URL")
    return value.rstrip("/")


def redact_text(value: Any) -> str:
    """Return provider text with common credential forms redacted."""

    text = str(value)
    text = _PRIVATE_KEY_BLOCK_RE.sub("[redacted]", text)
    text = re.sub(r"(?i)(Bearer\s+)[^\s,;]+", r"\1[redacted]", text)
    text = re.sub(r"(?i)((?:api[_-]?key|token|access_token|secret)=)[^&\s,;]+", r"\1[redacted]", text)
    text = re.sub(
        r"(?i)((?:authorization|x-api-key|x-subscription-token|api[_-]?key|token|access[_-]?token|secret):\s*)[^\r\n]+",
        r"\1[redacted]",
        text,
    )
    text = re.sub(r"sk-[A-Za-z0-9_-]{20,}", "[redacted]", text)
    text = re.sub(r"ghp_[A-Za-z0-9_]{20,}", "[redacted]", text)
    text = re.sub(r"AKIA[0-9A-Z]{16}", "[redacted]", text)
    text = re.sub(r"xox[bp]-[A-Za-z0-9-]{20,}", "[redacted]", text)
    return text


def _error_signature_text(value: Any, *, _depth: int = 0, _seen: Optional[set[int]] = None) -> str:
    """Return searchable error text including SDK class names and nested causes."""

    if value is None or _depth > 3:
        return ""
    if _seen is None:
        _seen = set()
    value_id = id(value)
    if value_id in _seen:
        return ""
    _seen.add(value_id)
    parts = [str(value or "")]
    cls = value.__class__
    parts.extend([cls.__name__, cls.__module__])
    for item in getattr(value, "args", ()):
        if item is value:
            continue
        parts.append(_error_signature_text(item, _depth=_depth + 1, _seen=_seen))
    for attr in ("__cause__", "__context__", "reason"):
        nested = getattr(value, attr, None)
        if nested is not None and nested is not value:
            parts.append(_error_signature_text(nested, _depth=_depth + 1, _seen=_seen))
    return " ".join(part for part in parts if part).lower()


def is_sensitive_content_error(value: Any) -> bool:
    """Return whether an upstream error is a sensitive-content rejection."""

    text = _error_signature_text(value)
    return any(marker in text for marker in _SENSITIVE_CONTENT_MARKERS)


def is_provider_timeout_error(value: Any) -> bool:
    """Return whether an upstream error is a timeout."""

    text = _error_signature_text(value)
    return any(marker in text for marker in _TIMEOUT_MARKERS)


def user_facing_provider_error(value: Any) -> str:
    """Map provider failures to safe, product-facing text."""

    if is_sensitive_content_error(value):
        return SENSITIVE_CONTENT_USER_MESSAGE
    if is_provider_timeout_error(value):
        return PROVIDER_TIMEOUT_USER_MESSAGE
    return redact_text(value)


def retry_policy_from_env() -> RetryPolicy:
    """Build the default provider retry policy from non-secret environment settings."""

    return RetryPolicy(
        max_retries=_env_int("DEEPRESEARCH_PROVIDER_MAX_RETRIES", 3),
        base_delay_s=_env_float("DEEPRESEARCH_PROVIDER_RETRY_BASE_DELAY_S", 0.5),
        max_delay_s=_env_float("DEEPRESEARCH_PROVIDER_RETRY_MAX_DELAY_S", 4.0),
    )


def default_json_transport(request: urllib.request.Request) -> JsonDict:
    """Perform a JSON HTTP request."""

    budget = _current_provider_budget()
    timeout_s = _provider_timeout_s()
    if budget is not None:
        budget.check("provider_json_connect")
        timeout_s = budget.call_timeout_s(timeout_s)
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            raw = response.read().decode("utf-8", errors="replace")
        data = json.loads(raw) if raw else {}
    except urllib.error.HTTPError as error:
        raise _provider_error_from_http(error) from error
    except (urllib.error.URLError, TimeoutError) as error:
        raise ProviderRequestError(error, retryable=True) from error
    except Exception as error:
        raise ProviderRequestError(error) from error
    if budget is not None:
        budget.check("provider_json_read")
    return data


def request_json_with_retries(
    transport: Transport,
    request: urllib.request.Request,
    *,
    retry_policy: RetryPolicy,
    sleeper: Sleeper = time.sleep,
) -> JsonDict:
    """Run one provider JSON request with bounded transient retry behavior."""

    retry_index = 0
    base_budget = _current_provider_budget()
    while True:
        attempt_budget = (
            base_budget.for_attempt(base_budget.attempt + retry_index)
            if base_budget is not None
            else None
        )
        if attempt_budget is not None:
            attempt_budget.check("provider_attempt")
        try:
            with _provider_budget_context(attempt_budget):
                return transport(request)
        except urllib.error.HTTPError as error:
            provider_error = _provider_error_from_http(error)
        except (urllib.error.URLError, TimeoutError) as error:
            provider_error = ProviderRequestError(error, retryable=True)
        except ProviderRequestError as error:
            provider_error = error

        if retry_index >= max(0, retry_policy.max_retries) or not _is_retryable_provider_error(provider_error, retry_policy):
            raise provider_error
        delay = retry_policy.delay_for(retry_index)
        if attempt_budget is not None:
            if not attempt_budget.retry_allowed(delay):
                # Distinguish attempt exhaustion (surface the provider failure) from
                # deadline exhaustion (surface stable RUN_TIMEOUT).
                if attempt_budget.attempt + 1 >= max(1, attempt_budget.max_attempts):
                    raise provider_error
                attempt_budget.check("provider_retry_backoff", reserve_s=delay)
            attempt_budget.sleep(delay, sleeper=sleeper)
        elif delay > 0:
            sleeper(delay)
        retry_index += 1


class LLMClient:
    """OpenAI-compatible chat completions client."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        transport: Optional[Transport] = None,
        retry_policy: Optional[RetryPolicy] = None,
        sleeper: Sleeper = time.sleep,
    ) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.transport = transport or default_json_transport
        self.retry_policy = retry_policy or retry_policy_from_env()
        self.sleeper = sleeper

    def require_ready(self) -> None:
        """Validate required BYOK LLM settings."""

        missing = []
        if not self.base_url.strip():
            missing.append("Base URL")
        if not self.api_key.strip():
            missing.append("API Key")
        if not self.model.strip():
            missing.append("Model")
        if missing:
            raise MissingCredentialError(f"BYOK LLM 未配置：请填写 {', '.join(missing)}。")
        validate_http_url(self.base_url, field_name="base_url")

    def chat(
        self,
        messages: List[JsonDict],
        *,
        max_tokens: int = 1200,
        temperature: float = 0.2,
        reasoning_effort: Optional[str] = None,
        response_format: Optional[JsonDict] = None,
    ) -> str:
        """Send a legacy chat completion request and return assistant content.

        ``reasoning_effort`` (``low``/``medium``/``high``) is forwarded only when
        set, so endpoints that do not support extended reasoning are unaffected.
        """

        return self.chat_result(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            response_format=response_format,
        ).text

    def chat_result(
        self,
        messages: List[JsonDict],
        *,
        max_tokens: int = 1200,
        temperature: float = 0.2,
        reasoning_effort: Optional[str] = None,
        response_format: Optional[JsonDict] = None,
        request_fields: Optional[JsonDict] = None,
    ) -> LLMChatResult:
        """Return text and terminal metadata from one Chat Completions call.

        ``request_fields`` is supplied only by the capability resolver.  Legacy
        callers can continue to use the explicit keyword arguments above.
        """

        self.require_ready()
        payload: JsonDict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if request_fields is not None:
            payload.update(dict(request_fields))
        else:
            payload["max_tokens"] = max_tokens
            if reasoning_effort:
                payload["reasoning_effort"] = reasoning_effort
            if response_format:
                payload["response_format"] = response_format
        request = urllib.request.Request(
            join_url(self.base_url, "/chat/completions"),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        data = request_json_with_retries(
            self.transport,
            request,
            retry_policy=self.retry_policy,
            sleeper=self.sleeper,
        )
        choices = data.get("choices", []) if isinstance(data, dict) else []
        choice = choices[0] if choices else {}
        message = choice.get("message", {}) if isinstance(choice, dict) else {}
        text = _chat_content_text(message.get("content") if isinstance(message, dict) else None)
        if not text and isinstance(choice, dict):
            text = _chat_content_text(choice.get("text"))
        finish_reason = (
            str(choice.get("finish_reason"))
            if isinstance(choice, dict) and choice.get("finish_reason") is not None
            else None
        )
        usage = dict(data.get("usage") or {}) if isinstance(data, dict) else {}
        response_model = str(data.get("model") or self.model) if isinstance(data, dict) else self.model
        return LLMChatResult(
            text=text,
            model=response_model,
            finish_reason=finish_reason,
            usage=usage,
            malformed_terminal=not (
                isinstance(choices, list)
                and bool(choices)
                and isinstance(choice, dict)
            ),
        )


def _chat_content_text(value: Any) -> str:
    """Normalize OpenAI-compatible string or typed-part message content."""

    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
            continue
        if not isinstance(item, dict):
            continue
        text = item.get("text") or item.get("content") or item.get("output_text")
        if isinstance(text, str):
            parts.append(text)
        elif isinstance(text, dict) and isinstance(text.get("value"), str):
            parts.append(text["value"])
    return "".join(parts)


def parse_json_object(text: str, fallback: JsonDict) -> JsonDict:
    """Parse a JSON object from LLM text with a fallback."""

    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else fallback
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            try:
                data = json.loads(text[start : end + 1])
                return data if isinstance(data, dict) else fallback
            except json.JSONDecodeError:
                return fallback
    return fallback


def _provider_error_from_http(error: urllib.error.HTTPError) -> ProviderRequestError:
    body = error.read().decode("utf-8", errors="replace")
    message = body or f"HTTP {error.code} {error.reason}"
    return ProviderRequestError(
        message,
        status_code=error.code,
        retryable=error.code not in _NEVER_RETRY_HTTP_STATUSES
        and (error.code in _RETRY_HTTP_STATUSES or _has_retry_marker(message)),
    )


def _is_retryable_provider_error(error: ProviderRequestError, policy: RetryPolicy) -> bool:
    status_code = error.status_code
    if status_code is not None and status_code in _NEVER_RETRY_HTTP_STATUSES:
        return False
    if status_code is not None and status_code in policy.retry_http_statuses:
        return True
    return bool(error.retryable or _has_retry_marker(str(error)))


def _has_retry_marker(value: Any) -> bool:
    text = str(value or "").lower()
    return any(marker in text for marker in _RETRY_TEXT_MARKERS)


def _provider_timeout_s() -> float:
    return _env_float("DEEPRESEARCH_PROVIDER_TIMEOUT_S", 60.0, minimum=0.1)


def _env_int(name: str, default: int) -> int:
    try:
        return max(0, int(str(os.getenv(name, default)).strip()))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(str(os.getenv(name, default)).strip()))
    except (TypeError, ValueError):
        return default
