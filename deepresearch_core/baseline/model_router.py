"""Model Router: selects an LLM backend (LiteLLM / Ollama / OpenAI-compatible).

The router itself satisfies :class:`LLMAdapter`. Backends are lazily imported, so the
absence of ``litellm`` or a running Ollama simply selects the next available backend.
BYOK precedence is honored by the caller-supplied :class:`ProviderSettings` /
:class:`LLMClient`; keys are never logged.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from collections.abc import Iterator
from typing import Any, Sequence

from ..settings import ProviderSettings
from ..standalone.llm import LLMClient, redact_text
from .adapters import (
    LLMAdapter,
    LLMResult,
    LLMStreamChunk,
    OpenAICompatLLMAdapter,
    classify_llm_terminal,
)
from .cost_estimate import estimate_llm_cost
from .deadline import ProviderCallBudget, current_provider_call_budget, use_provider_call_budget
from .errors import BaselineError, ErrorCode
from .llm_call_params import resolve_llm_call_options

JsonDict = dict[str, Any]


@dataclass(frozen=True)
class ModelSpec:
    """Identifies a backend + model name (used for council fan-out)."""

    backend: str
    model: str


class OllamaLLMAdapter:
    """LLMAdapter for a local Ollama server (chat + embeddings)."""

    name = "ollama"

    def __init__(self, *, base_url: str = "http://127.0.0.1:11434", model: str = "llama3.1", timeout_s: float = 60.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def available(self) -> bool:
        """Return whether the local Ollama server exposes the selected model."""

        try:
            payload = self._get("/api/tags")
        except Exception:  # noqa: BLE001 - availability probe must never raise
            return False
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            return False
        expected = {self.model}
        if ":" not in self.model:
            expected.add(f"{self.model}:latest")
        elif self.model.endswith(":latest"):
            expected.add(self.model.removesuffix(":latest"))
        return any(
            str(item.get("name") or item.get("model") or "").strip() in expected
            for item in models
            if isinstance(item, dict)
        )

    def complete(
        self,
        messages: Sequence[JsonDict],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        thinking_budget: int | None = None,  # noqa: ARG002 - Ollama has no thinking dial
        response_format: JsonDict | None = None,  # noqa: ARG002 - Ollama has no response_format dial
    ) -> LLMResult:
        """Complete via Ollama's ``/api/chat`` endpoint (thinking budget ignored)."""

        payload = {
            "model": model or self.model,
            "messages": list(messages),
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        data = self._post("/api/chat", payload)
        text = str((data.get("message") or {}).get("content") or "")
        finish_reason = str(data.get("done_reason") or "stop") if data.get("done") else None
        usage = {
            key: data[key]
            for key in ("prompt_eval_count", "eval_count")
            if key in data
        }
        return LLMResult(
            text=text,
            model=model or self.model,
            finish_reason=finish_reason,
            usage=usage,
            diagnostics={
                "classification": classify_llm_terminal(
                    text=text,
                    finish_reason=finish_reason,
                    usage=usage,
                ),
                "call_options": {
                    "provider_identity": "ollama",
                    "provider_family": "ollama",
                    "endpoint_type": "ollama_chat",
                    "reasoning_transport": "none",
                    "structured_output_strategy": "prompt_validated",
                    "output_token_parameter": "num_predict",
                    "shared_output_cap": False,
                    "provider_output_cap": max_tokens,
                    "requested": {
                        "thinking_effort": "off",
                        "visible_output_tokens": max_tokens,
                        "structured_output": bool(response_format),
                    },
                    "applied": {"visible_output_limit": "num_predict"},
                    "ignored": {},
                    "reason_codes": [],
                },
                "visible_chars": len(text),
            },
        )

    def stream(
        self,
        messages: Sequence[JsonDict],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        thinking_budget: int | None = None,  # noqa: ARG002
        response_format: JsonDict | None = None,  # noqa: ARG002
    ) -> Iterator[LLMStreamChunk]:
        """Stream Ollama's newline-delimited chat response."""

        selected_model = model or self.model
        payload = {
            "model": selected_model,
            "messages": list(messages),
            "stream": True,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        request = urllib.request.Request(
            self.base_url + "/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        budget = current_provider_call_budget()
        timeout_s = self.timeout_s
        if budget is not None:
            budget.check("ollama_stream_connect")
            timeout_s = budget.call_timeout_s(timeout_s)
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                terminal_emitted = False
                visible_chars = 0
                for raw_line in response:
                    if budget is not None:
                        budget.check("ollama_stream_read")
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    delta = str((data.get("message") or {}).get("content") or "")
                    done = bool(data.get("done"))
                    finish_reason = str(data.get("done_reason") or "stop") if done else None
                    usage = {
                        key: data[key]
                        for key in ("prompt_eval_count", "eval_count")
                        if key in data
                    }
                    yield LLMStreamChunk(
                        delta=delta,
                        done=done,
                        model=selected_model,
                        provider=self.name,
                        usage=usage,
                        finish_reason=finish_reason,
                        diagnostics=(
                            {
                                "classification": classify_llm_terminal(
                                    text="x" * (visible_chars + len(delta)),
                                    finish_reason=finish_reason,
                                    usage=usage,
                                ),
                                "visible_chars": visible_chars + len(delta),
                            }
                            if done
                            else {}
                        ),
                    )
                    visible_chars += len(delta)
                    terminal_emitted = terminal_emitted or done
                if budget is not None:
                    budget.check("ollama_stream_read")
                if not terminal_emitted:
                    yield LLMStreamChunk(
                        done=True,
                        model=selected_model,
                        provider=self.name,
                        diagnostics={
                            "classification": classify_llm_terminal(
                                text="x" * visible_chars,
                                finish_reason=None,
                                usage={},
                            ),
                            "visible_chars": visible_chars,
                        },
                    )
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            raise BaselineError(ErrorCode.UPSTREAM_ERROR, redact_text(f"ollama stream error: {error}")) from error

    def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        """Embed via Ollama's ``/api/embeddings`` endpoint (one call per text)."""

        vectors: list[list[float]] = []
        for text in texts:
            data = self._post("/api/embeddings", {"model": model or self.model, "prompt": text})
            vector = data.get("embedding")
            if not isinstance(vector, list):
                raise BaselineError(ErrorCode.UPSTREAM_ERROR, "ollama returned no embedding")
            vectors.append([float(x) for x in vector])
        return vectors

    def estimate_cost(
        self,
        *,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> JsonDict:
        """Return a local-pricing placeholder without contacting Ollama."""

        low, high, basis, disclaimer = estimate_llm_cost(
            model_id=model or self.model,
            provider_type="ollama",
            input_tokens=max(0, int(input_tokens)),
            output_tokens=max(0, int(output_tokens)),
        )
        return {"usd_low": low, "usd_high": high, "basis": basis, "disclaimer": disclaimer}

    def _post(self, path: str, payload: JsonDict) -> JsonDict:
        request = urllib.request.Request(
            self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return self._send(request)

    def _get(self, path: str) -> JsonDict:
        return self._send(urllib.request.Request(self.base_url + path, method="GET"))

    def _send(self, request: urllib.request.Request) -> JsonDict:
        budget = current_provider_call_budget()
        timeout_s = self.timeout_s
        if budget is not None:
            budget.check("ollama_json_connect")
            timeout_s = budget.call_timeout_s(timeout_s)
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                raw = response.read().decode("utf-8", errors="replace")
            data = json.loads(raw) if raw else {}
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            raise BaselineError(ErrorCode.UPSTREAM_ERROR, redact_text(f"ollama error: {error}")) from error
        if budget is not None:
            budget.check("ollama_json_read")
        return data


class LiteLLMAdapter:
    """LLMAdapter backed by the optional ``litellm`` package (multi-provider)."""

    name = "litellm"

    def __init__(self, *, model: str, api_key: str = "", base_url: str = "") -> None:
        self.model = model
        self._api_key = api_key
        self._base_url = base_url

    @staticmethod
    def available() -> bool:
        """Return whether ``litellm`` can be imported."""

        try:
            import litellm  # noqa: F401
            return True
        except Exception:  # noqa: BLE001
            return False

    def complete(
        self,
        messages: Sequence[JsonDict],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        thinking_budget: int | None = None,
        response_format: JsonDict | None = None,
    ) -> LLMResult:
        """Complete via ``litellm.completion``."""

        try:
            import litellm
        except Exception as error:  # noqa: BLE001
            raise BaselineError(ErrorCode.PROVIDER_NOT_CONFIGURED, "litellm is not installed") from error
        selected_model = model or self.model
        call_options = resolve_llm_call_options(
            provider_type="litellm",
            provider_id="litellm",
            base_url=self._base_url,
            model=selected_model,
            thinking_budget=thinking_budget,
            max_visible_tokens=max_tokens,
            response_format=response_format,
        )
        kwargs: JsonDict = {
            "model": selected_model,
            "messages": list(messages),
            "temperature": temperature,
            **call_options.request_fields,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["api_base"] = self._base_url
        budget = current_provider_call_budget()
        if budget is not None:
            budget.check("litellm_complete")
            kwargs["timeout"] = budget.call_timeout_s(60.0)
            kwargs["max_retries"] = 0
        try:
            response = litellm.completion(**kwargs)
        except Exception as error:  # noqa: BLE001
            if isinstance(error, BaselineError):
                raise
            raise BaselineError(ErrorCode.UPSTREAM_ERROR, redact_text(f"litellm error: {error}")) from error
        if budget is not None:
            budget.check("litellm_complete")
        choice = _first_choice(response)
        message = _value(choice, "message") or {}
        text = str(_value(message, "content") or _value(choice, "text") or "")
        finish_reason = _value(choice, "finish_reason")
        usage = _mapping(_value(response, "usage"))
        return LLMResult(
            text=text,
            model=str(_value(response, "model") or selected_model),
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            usage=usage,
            diagnostics={
                "classification": classify_llm_terminal(
                    text=text,
                    finish_reason=str(finish_reason) if finish_reason is not None else None,
                    usage=usage,
                ),
                "call_options": call_options.to_dict(),
                "visible_chars": len(text),
            },
        )

    def stream(
        self,
        messages: Sequence[JsonDict],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        thinking_budget: int | None = None,
        response_format: JsonDict | None = None,
    ) -> Iterator[LLMStreamChunk]:
        """Yield normalized chunks from LiteLLM's streaming iterator."""

        try:
            import litellm
        except Exception as error:  # noqa: BLE001
            raise BaselineError(ErrorCode.PROVIDER_NOT_CONFIGURED, "litellm is not installed") from error
        selected_model = model or self.model
        call_options = resolve_llm_call_options(
            provider_type="litellm",
            provider_id="litellm",
            base_url=self._base_url,
            model=selected_model,
            thinking_budget=thinking_budget,
            max_visible_tokens=max_tokens,
            response_format=response_format,
        )
        kwargs: JsonDict = {
            "model": selected_model,
            "messages": list(messages),
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
            **call_options.request_fields,
        }
        if self._api_key:
            kwargs["api_key"] = self._api_key
        if self._base_url:
            kwargs["api_base"] = self._base_url
        budget = current_provider_call_budget()
        if budget is not None:
            budget.check("litellm_stream")
            kwargs["timeout"] = budget.call_timeout_s(60.0)
            kwargs["max_retries"] = 0
        try:
            response = litellm.completion(**kwargs)
            terminal_emitted = False
            visible_chars = 0
            final_usage: JsonDict = {}
            final_finish_reason: str | None = None
            for raw in response:
                if budget is not None:
                    budget.check("litellm_stream")
                choice = _first_choice(raw)
                delta_obj = _value(choice, "delta") or {}
                delta = str(_value(delta_obj, "content") or _value(choice, "text") or "")
                finish_reason = _value(choice, "finish_reason")
                usage = _mapping(_value(raw, "usage"))
                done = finish_reason is not None
                visible_chars += len(delta)
                if usage:
                    final_usage.update(usage)
                if finish_reason is not None:
                    final_finish_reason = str(finish_reason)
                yield LLMStreamChunk(
                    delta=delta,
                    done=done,
                    model=str(_value(raw, "model") or selected_model),
                    provider=self.name,
                    usage=usage,
                    finish_reason=final_finish_reason if done else None,
                    diagnostics=(
                        {
                            "classification": classify_llm_terminal(
                                text="x" * visible_chars,
                                finish_reason=final_finish_reason,
                                usage=final_usage,
                            ),
                            "call_options": call_options.to_dict(),
                            "visible_chars": visible_chars,
                        }
                        if done
                        else {}
                    ),
                )
                terminal_emitted = terminal_emitted or done
            if not terminal_emitted:
                yield LLMStreamChunk(
                    done=True,
                    model=selected_model,
                    provider=self.name,
                    usage=final_usage,
                    finish_reason=final_finish_reason,
                    diagnostics={
                        "classification": classify_llm_terminal(
                            text="x" * visible_chars,
                            finish_reason=final_finish_reason,
                            usage=final_usage,
                        ),
                        "call_options": call_options.to_dict(),
                        "visible_chars": visible_chars,
                    },
                )
        except Exception as error:  # noqa: BLE001
            if isinstance(error, BaselineError):
                raise
            raise BaselineError(ErrorCode.UPSTREAM_ERROR, redact_text(f"litellm stream error: {error}")) from error

    def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        """Embed via ``litellm.embedding``."""

        try:
            import litellm
        except Exception as error:  # noqa: BLE001
            raise BaselineError(ErrorCode.PROVIDER_NOT_CONFIGURED, "litellm is not installed") from error
        try:
            kwargs: JsonDict = {"model": model or self.model, "input": list(texts)}
            budget = current_provider_call_budget()
            if budget is not None:
                budget.check("litellm_embed")
                kwargs["timeout"] = budget.call_timeout_s(60.0)
                kwargs["max_retries"] = 0
            response = litellm.embedding(**kwargs)
        except Exception as error:  # noqa: BLE001
            if isinstance(error, BaselineError):
                raise
            raise BaselineError(ErrorCode.UPSTREAM_ERROR, redact_text(f"litellm embed error: {error}")) from error
        if budget is not None:
            budget.check("litellm_embed")
        return [[float(x) for x in item["embedding"]] for item in response["data"]]

    def estimate_cost(
        self,
        *,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> JsonDict:
        """Estimate completion cost using LiteLLM pricing metadata when available."""

        low, high, basis, disclaimer = estimate_llm_cost(
            model_id=model or self.model,
            provider_type="litellm",
            input_tokens=max(0, int(input_tokens)),
            output_tokens=max(0, int(output_tokens)),
        )
        return {"usd_low": low, "usd_high": high, "basis": basis, "disclaimer": disclaimer}


class ModelRouter:
    """Routes completion/embedding to the first healthy backend; satisfies LLMAdapter."""

    name = "model-router"
    owns_fallback = True

    def __init__(self, backends: Sequence[LLMAdapter]) -> None:
        if not backends:
            raise BaselineError(ErrorCode.PROVIDER_NOT_CONFIGURED, "no LLM backend configured")
        self._backends = list(backends)

    @property
    def backends(self) -> list[LLMAdapter]:
        """Return the ordered backend list."""

        return list(self._backends)

    def complete(
        self,
        messages: Sequence[JsonDict],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        thinking_budget: int | None = None,
        response_format: JsonDict | None = None,
    ) -> LLMResult:
        """Try each backend until one succeeds; raise the last error otherwise."""

        last_error: Exception | None = None
        attempts: list[JsonDict] = []
        base_budget = current_provider_call_budget()
        for backend_index, backend in enumerate(self._backends):
            attempt = (base_budget.attempt if base_budget is not None else 0) + backend_index
            if base_budget is not None and attempt >= max(1, base_budget.max_attempts):
                break
            attempt_budget = _router_attempt_budget(base_budget, attempt)
            if attempt_budget is not None:
                attempt_budget.check("model_router_complete")
            try:
                with use_provider_call_budget(attempt_budget):
                    result = backend.complete(
                        messages,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        thinking_budget=thinking_budget,
                        response_format=response_format,
                    )
                classification = _result_classification(result)
                attempts.append(
                    _attempt_record(
                        attempt=attempt,
                        backend=backend,
                        call_mode="sync",
                        classification=classification,
                        status="completed" if result.text.strip() else "empty",
                        fallback_scope="cross_backend" if backend_index else "none",
                        finish_reason=result.finish_reason,
                        usage=result.usage,
                        visible_chars=len(result.text),
                        call_options=result.diagnostics.get("call_options"),
                        provider=_result_provider(result, backend),
                        model=result.model,
                    )
                )
                if result.text.strip():
                    return LLMResult(
                        text=result.text,
                        model=result.model,
                        finish_reason=result.finish_reason,
                        usage=dict(result.usage),
                        diagnostics={
                            **dict(result.diagnostics),
                            "classification": classification,
                            "fallback_scope": "cross_backend" if backend_index else "none",
                            "attempts": list(attempts),
                        },
                    )
                last_error = BaselineError(
                    ErrorCode.UPSTREAM_ERROR,
                    "LLM provider returned an empty answer.",
                    details={
                        "writer_diagnostics": _writer_diagnostics(
                            attempts,
                            classification=classification,
                        )
                    },
                )
            except Exception as error:  # noqa: BLE001 - fall through to next backend
                if isinstance(error, BaselineError) and error.code is ErrorCode.RUN_TIMEOUT:
                    raise
                attempts.append(
                    _attempt_record(
                        attempt=attempt,
                        backend=backend,
                        call_mode="sync",
                        classification=_error_classification(error),
                        status="failed",
                        fallback_scope="cross_backend" if backend_index else "none",
                    )
                )
                last_error = error
        raise _router_error(last_error, attempts)

    def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        """Embed via the first backend that supports it."""

        last_error: Exception | None = None
        base_budget = current_provider_call_budget()
        for backend_index, backend in enumerate(self._backends):
            attempt = (base_budget.attempt if base_budget is not None else 0) + backend_index
            if base_budget is not None and attempt >= max(1, base_budget.max_attempts):
                break
            attempt_budget = _router_attempt_budget(base_budget, attempt)
            if attempt_budget is not None:
                attempt_budget.check("model_router_embed")
            try:
                with use_provider_call_budget(attempt_budget):
                    return backend.embed(texts, model=model)
            except Exception as error:  # noqa: BLE001
                if isinstance(error, BaselineError) and error.code is ErrorCode.RUN_TIMEOUT:
                    raise
                last_error = error
        raise _as_baseline_error(last_error)

    def stream(
        self,
        messages: Sequence[JsonDict],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
        thinking_budget: int | None = None,
        response_format: JsonDict | None = None,
    ) -> Iterator[LLMStreamChunk]:
        """Stream from the first viable backend, falling back only before output."""

        kwargs = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "thinking_budget": thinking_budget,
            "response_format": response_format,
        }
        last_error: Exception | None = None
        attempts: list[JsonDict] = []
        base_budget = current_provider_call_budget()
        call_index = 0
        for backend_index, backend in enumerate(self._backends):
            stream_fn = getattr(backend, "stream", None)
            if callable(stream_fn):
                attempt = (base_budget.attempt if base_budget is not None else 0) + call_index
                if base_budget is not None and attempt >= max(1, base_budget.max_attempts):
                    break
                call_index += 1
                attempt_budget = _router_attempt_budget(base_budget, attempt)
                if attempt_budget is not None:
                    attempt_budget.check("model_router_stream")
                emitted = False
                terminal = False
                iterator = None
                terminal_classification = "empty_stream"
                last_observed = LLMStreamChunk()
                stream_usage: JsonDict = {}
                stream_visible_chars = 0
                stream_finish_reason: str | None = None
                stream_call_options: JsonDict = {}
                stream_provider: str | None = None
                stream_model: str | None = None
                try:
                    with use_provider_call_budget(attempt_budget):
                        iterator = iter(stream_fn(messages, **kwargs))
                    while True:
                        try:
                            if attempt_budget is not None:
                                attempt_budget.check("model_router_stream")
                            with use_provider_call_budget(attempt_budget):
                                chunk = next(iterator)
                        except StopIteration:
                            break
                        normalized = _stream_chunk(chunk, backend=backend, model=model)
                        if normalized.done or normalized.usage or normalized.diagnostics:
                            last_observed = normalized
                        if normalized.delta:
                            emitted = True
                            stream_visible_chars += len(normalized.delta)
                        if normalized.usage:
                            stream_usage.update(normalized.usage)
                        if normalized.finish_reason is not None:
                            stream_finish_reason = normalized.finish_reason
                        if normalized.provider:
                            stream_provider = normalized.provider
                        if normalized.model:
                            stream_model = normalized.model
                        raw_options = normalized.diagnostics.get("call_options")
                        if isinstance(raw_options, dict):
                            stream_call_options = dict(raw_options)
                        if normalized.diagnostics.get("classification"):
                            terminal_classification = str(normalized.diagnostics["classification"])
                        elif normalized.done:
                            terminal_classification = classify_llm_terminal(
                                text="x" if emitted else "",
                                finish_reason=normalized.finish_reason,
                                usage=normalized.usage,
                            )
                        if normalized.delta or normalized.done or normalized.usage or normalized.diagnostics:
                            yield _chunk_with_router_diagnostics(
                                normalized,
                                attempt=attempt,
                                backend=backend,
                                fallback_scope="cross_backend" if backend_index else "none",
                            )
                        terminal = terminal or (normalized.done and emitted)
                    if emitted:
                        terminal_classification = classify_llm_terminal(
                            text="x" * stream_visible_chars,
                            finish_reason=stream_finish_reason,
                            usage=stream_usage,
                        )
                        attempts.append(
                            _attempt_record(
                                attempt=attempt,
                                backend=backend,
                                call_mode="stream",
                                classification=terminal_classification,
                                status="completed",
                                fallback_scope="cross_backend" if backend_index else "none",
                                finish_reason=stream_finish_reason,
                                usage=stream_usage,
                                visible_chars=stream_visible_chars,
                                call_options=stream_call_options,
                                provider=stream_provider,
                                model=stream_model,
                            )
                        )
                        if not terminal:
                            yield LLMStreamChunk(
                                done=True,
                                model=model or getattr(backend, "model", None),
                                provider=str(getattr(backend, "name", "")) or None,
                                diagnostics={
                                    "classification": terminal_classification,
                                    "attempt": attempt,
                                    "fallback_scope": "cross_backend" if backend_index else "none",
                                    "attempts": list(attempts),
                                },
                            )
                        else:
                            yield LLMStreamChunk(
                                done=True,
                                model=stream_model or model or getattr(backend, "model", None),
                                provider=stream_provider
                                or str(getattr(backend, "name", ""))
                                or None,
                                usage=dict(stream_usage),
                                finish_reason=stream_finish_reason,
                                diagnostics={
                                    **dict(last_observed.diagnostics),
                                    "classification": terminal_classification,
                                    "attempt": attempt,
                                    "fallback_scope": "cross_backend" if backend_index else "none",
                                    "attempts": list(attempts),
                                },
                            )
                        return
                    attempts.append(
                        _attempt_record(
                            attempt=attempt,
                            backend=backend,
                            call_mode="stream",
                            classification=terminal_classification,
                            status="empty",
                            fallback_scope="cross_backend" if backend_index else "none",
                            finish_reason=stream_finish_reason,
                            usage=stream_usage,
                            visible_chars=stream_visible_chars,
                            call_options=stream_call_options,
                            provider=stream_provider,
                            model=stream_model,
                        )
                    )
                except Exception as error:  # noqa: BLE001
                    if isinstance(error, BaselineError) and error.code is ErrorCode.RUN_TIMEOUT:
                        raise
                    if emitted:
                        attempts.append(
                            _attempt_record(
                                attempt=attempt,
                                backend=backend,
                                call_mode="stream",
                                classification=_error_classification(error),
                                status="failed",
                                fallback_scope="cross_backend" if backend_index else "none",
                                finish_reason=stream_finish_reason,
                                usage=stream_usage,
                                visible_chars=stream_visible_chars,
                                call_options=stream_call_options,
                                provider=stream_provider,
                                model=stream_model,
                            )
                        )
                        raise _router_error(error, attempts) from error
                    attempts.append(
                        _attempt_record(
                            attempt=attempt,
                            backend=backend,
                            call_mode="stream",
                            classification=_error_classification(error),
                            status="failed",
                            fallback_scope="cross_backend" if backend_index else "none",
                        )
                    )
                    last_error = error
                finally:
                    close = getattr(iterator, "close", None)
                    if callable(close):
                        close()
            attempt = (base_budget.attempt if base_budget is not None else 0) + call_index
            if base_budget is not None and attempt >= max(1, base_budget.max_attempts):
                break
            call_index += 1
            attempt_budget = _router_attempt_budget(base_budget, attempt)
            if attempt_budget is not None:
                attempt_budget.check("model_router_sync_fallback")
            fallback_scope = (
                "cross_backend"
                if backend_index
                else "same_provider_sync"
                if callable(stream_fn)
                else "sync_only"
            )
            try:
                with use_provider_call_budget(attempt_budget):
                    result = backend.complete(messages, **kwargs)
            except Exception as error:  # noqa: BLE001
                if isinstance(error, BaselineError) and error.code is ErrorCode.RUN_TIMEOUT:
                    raise
                attempts.append(
                    _attempt_record(
                        attempt=attempt,
                        backend=backend,
                        call_mode="sync",
                        classification=_error_classification(error),
                        status="failed",
                        fallback_scope=fallback_scope,
                    )
                )
                last_error = error
                continue
            classification = _result_classification(result)
            attempts.append(
                _attempt_record(
                    attempt=attempt,
                    backend=backend,
                    call_mode="sync",
                    classification=classification,
                    status="completed" if result.text.strip() else "empty",
                    fallback_scope=fallback_scope,
                    finish_reason=result.finish_reason,
                    usage=result.usage,
                    visible_chars=len(result.text),
                    call_options=result.diagnostics.get("call_options"),
                    provider=_result_provider(result, backend),
                    model=result.model,
                )
            )
            if result.text.strip():
                result_provider = _result_provider(result, backend)
                yield LLMStreamChunk(
                    delta=result.text,
                    model=result.model,
                    provider=result_provider,
                    diagnostics={
                        **dict(result.diagnostics),
                        "classification": classification,
                        "attempt": attempt,
                        "fallback_scope": fallback_scope,
                    },
                )
                yield LLMStreamChunk(
                    done=True,
                    model=result.model,
                    provider=result_provider,
                    usage={**result.usage, "sync_fallback": True},
                    finish_reason=result.finish_reason,
                    diagnostics={
                        **dict(result.diagnostics),
                        "classification": classification,
                        "attempt": attempt,
                        "fallback_scope": fallback_scope,
                        "attempts": list(attempts),
                    },
                )
                return
            last_error = BaselineError(
                ErrorCode.UPSTREAM_ERROR,
                "LLM provider returned an empty answer.",
                details={
                    "writer_diagnostics": _writer_diagnostics(
                        attempts,
                        classification=classification,
                    )
                },
            )
        raise _router_error(last_error, attempts)

    def estimate_cost(
        self,
        *,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> JsonDict:
        """Estimate via the first backend exposing a usable estimate."""

        for backend in self._backends:
            estimate = getattr(backend, "estimate_cost", None)
            if not callable(estimate):
                continue
            result = estimate(model=model, input_tokens=input_tokens, output_tokens=output_tokens)
            if isinstance(result, dict):
                return result
        return {"usd_low": None, "usd_high": None, "basis": {"source": "unknown"}, "disclaimer": "LLM pricing is unavailable for this model."}


def build_model_router(
    provider: ProviderSettings,
    *,
    llm_client: LLMClient | None = None,
    prefer_local: bool = False,
    ollama_base_url: str = "http://127.0.0.1:11434",
    ollama_model: str = "llama3.1",
) -> ModelRouter:
    """Build a router from BYOK provider settings with graceful backend selection.

    Order: optional local Ollama first when ``prefer_local`` and reachable, then the
    BYOK OpenAI-compatible endpoint, then LiteLLM when installed.
    """

    backends: list[LLMAdapter] = []
    ollama = OllamaLLMAdapter(base_url=ollama_base_url, model=ollama_model)
    if prefer_local and ollama.available():
        backends.append(ollama)

    if provider.llm_base_url and provider.llm_api_key and provider.llm_model:
        client = llm_client or LLMClient(
            base_url=provider.llm_base_url,
            api_key=provider.llm_api_key,
            model=provider.llm_model,
        )
        backends.append(OpenAICompatLLMAdapter(client))

    if not prefer_local and ollama.available():
        backends.append(ollama)

    if provider.llm_model and LiteLLMAdapter.available():
        backends.append(LiteLLMAdapter(model=provider.llm_model, api_key=provider.llm_api_key, base_url=provider.llm_base_url))

    if not backends:
        raise BaselineError(
            ErrorCode.PROVIDER_NOT_CONFIGURED,
            "No LLM backend available: configure BYOK provider or start a local Ollama server.",
        )
    return ModelRouter(backends)


def _as_baseline_error(error: Exception | None) -> BaselineError:
    if isinstance(error, BaselineError):
        return error
    return BaselineError(ErrorCode.UPSTREAM_ERROR, redact_text(str(error or "all LLM backends failed")))


def _router_error(error: Exception | None, attempts: list[JsonDict]) -> BaselineError:
    if isinstance(error, BaselineError) and error.code is ErrorCode.RUN_TIMEOUT:
        return error
    classification = (
        str(attempts[-1].get("classification") or "empty_stream")
        if attempts
        else _error_classification(error)
    )
    message = (
        error.message
        if isinstance(error, BaselineError)
        else redact_text(str(error or "all LLM backends failed"))
    )
    return BaselineError(
        ErrorCode.UPSTREAM_ERROR,
        message,
        details={
            "writer_diagnostics": _writer_diagnostics(
                attempts,
                classification=classification,
            )
        },
    )


def _writer_diagnostics(attempts: list[JsonDict], *, classification: str) -> JsonDict:
    final = attempts[-1] if attempts else {}
    return {
        "classification": classification,
        "attempts": list(attempts),
        "provider": final.get("provider"),
        "model": final.get("model"),
        "finish_reason": final.get("finish_reason"),
        "visible_chars": int(final.get("visible_chars") or 0),
        "usage": dict(final.get("usage") or {}),
        "fallback_scope": final.get("fallback_scope") or "none",
        "call_options": dict(final.get("call_options") or {}),
    }


def _router_attempt_budget(
    base_budget: ProviderCallBudget | None,
    attempt: int,
) -> ProviderCallBudget | None:
    """Allocate exactly one global attempt to a backend invocation.

    Provider transports retain standalone retries, but a call made through the
    router must not spend the same immutable attempt slot internally and then let
    the router reuse it for another backend.
    """

    if base_budget is None:
        return None
    current = base_budget.for_attempt(attempt)
    return replace(current, max_attempts=max(1, attempt + 1))


def _attempt_record(
    *,
    attempt: int,
    backend: LLMAdapter,
    call_mode: str,
    classification: str,
    status: str,
    fallback_scope: str,
    finish_reason: str | None = None,
    usage: JsonDict | None = None,
    visible_chars: int = 0,
    call_options: Any = None,
    provider: str | None = None,
    model: str | None = None,
) -> JsonDict:
    return {
        "attempt": max(0, int(attempt)),
        "provider": provider or str(getattr(backend, "name", "") or "unknown"),
        "model": model or str(getattr(backend, "model", "") or "") or None,
        "call_mode": call_mode,
        "classification": classification,
        "status": status,
        "fallback_scope": fallback_scope,
        "finish_reason": finish_reason,
        "usage": dict(usage or {}),
        "visible_chars": max(0, int(visible_chars)),
        "call_options": dict(call_options) if isinstance(call_options, dict) else {},
    }


def _result_classification(result: LLMResult) -> str:
    return str(result.diagnostics.get("classification") or classify_llm_terminal(
        text=result.text,
        finish_reason=result.finish_reason,
        usage=result.usage,
    ))


def _result_provider(result: LLMResult, backend: LLMAdapter) -> str:
    call_options = result.diagnostics.get("call_options")
    if isinstance(call_options, dict) and call_options.get("provider_identity"):
        return str(call_options["provider_identity"])
    return str(getattr(backend, "name", "") or "unknown")


def _error_classification(error: Exception | None) -> str:
    if isinstance(error, BaselineError):
        if error.code is ErrorCode.RUN_TIMEOUT:
            return "deadline_exceeded"
        diagnostics = error.details.get("writer_diagnostics")
        if isinstance(diagnostics, dict) and diagnostics.get("classification"):
            return str(diagnostics["classification"])
    return "provider_error"


def _chunk_with_router_diagnostics(
    chunk: LLMStreamChunk,
    *,
    attempt: int,
    backend: LLMAdapter,
    fallback_scope: str,
) -> LLMStreamChunk:
    return LLMStreamChunk(
        delta=chunk.delta,
        done=chunk.done,
        model=chunk.model,
        provider=chunk.provider or str(getattr(backend, "name", "") or "") or None,
        usage=dict(chunk.usage),
        finish_reason=chunk.finish_reason,
        diagnostics={
            **dict(chunk.diagnostics),
            "attempt": max(0, int(attempt)),
            "fallback_scope": fallback_scope,
        },
    )


def _value(value: Any, key: str) -> Any:
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


def _mapping(value: Any) -> JsonDict:
    if isinstance(value, dict):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dict(dumped) if isinstance(dumped, dict) else {}
    return {}


def _first_choice(value: Any) -> Any:
    choices = _value(value, "choices") or []
    return choices[0] if choices else {}


def _stream_chunk(value: Any, *, backend: LLMAdapter, model: str | None) -> LLMStreamChunk:
    if isinstance(value, LLMStreamChunk):
        return value
    return LLMStreamChunk(
        delta=str(_value(value, "delta") or ""),
        done=bool(_value(value, "done")),
        model=str(_value(value, "model") or model or getattr(backend, "model", "")) or None,
        provider=str(_value(value, "provider") or getattr(backend, "name", "")) or None,
        usage=_mapping(_value(value, "usage")),
        finish_reason=(
            str(_value(value, "finish_reason"))
            if _value(value, "finish_reason") is not None
            else None
        ),
        diagnostics=_mapping(_value(value, "diagnostics")),
    )
