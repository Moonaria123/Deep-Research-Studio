"""Provider adapter contracts and concrete implementations.

Defines the ``LLMAdapter`` and ``SearchToolAdapter`` protocols (baseline §接口契约)
and ships standard-library implementations that wrap the existing OpenAI-compatible
LLM client and configurable Brave/Tavily/Exa/Firecrawl search clients. Optional backends (Ollama, LiteLLM)
live in :mod:`deepresearch_core.baseline.model_router`.
"""

from __future__ import annotations

import json
import hashlib
import inspect
import math
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections.abc import Iterator
from typing import Any, Protocol, Sequence, runtime_checkable

from ..standalone.llm import (
    LLMClient,
    ProviderRequestError,
    is_provider_timeout_error,
    join_url,
    redact_text,
    request_json_with_retries,
    retry_policy_from_env,
)
from ..standalone.search import SearchClient, tavily_country_name
from .errors import BaselineError, ErrorCode
from .cost_estimate import estimate_llm_cost, estimate_search_cost
from .deadline import current_provider_call_budget, use_provider_call_budget
from .llm_call_params import resolve_llm_call_options
from .search_options import SearchOptions

JsonDict = dict[str, Any]

_EXA_EXTENDED_TEXT_FOCUS = re.compile(
    r"\b(?:http\s+status|status(?:\s+code)?|error\s+code|rfc|iso|iec|section)"
    r"\s*[:#-]?\s*\d{3,6}\b",
    re.IGNORECASE,
)
_EXA_EXTENDED_TEXT_MAX_CHARACTERS = 500_000
_EXA_FOCUSED_BODY_WINDOW_CHARACTERS = 12_000
_EXA_DEFINITION_CONTEXT = re.compile(
    r"\b(?:means?|indicates?|defined|definition|not\s+found|origin\s+server|"
    r"representation|status\s+code)\b",
    re.IGNORECASE,
)


def _focused_exa_numeric_body_excerpt(
    result: JsonDict,
    *,
    relevance_query: str,
) -> str:
    """Recover a bounded target window when Exa highlights miss a numeric standard."""

    focus_match = _EXA_EXTENDED_TEXT_FOCUS.search(relevance_query)
    token_match = re.search(r"\d{3,6}", focus_match.group(0)) if focus_match else None
    raw = result.get("raw") if isinstance(result, dict) else None
    body = str(
        (raw.get("text") or raw.get("content") or "")
        if isinstance(raw, dict)
        else ""
    ).strip()
    if token_match is None or not body:
        return ""

    token = token_match.group(0)
    occurrences = tuple(
        match.start()
        for match in re.finditer(rf"(?<!\d){re.escape(token)}(?!\d)", body)
    )
    if not occurrences:
        return ""

    radius = _EXA_FOCUSED_BODY_WINDOW_CHARACTERS // 2
    best: tuple[int, int, str] | None = None
    for position in occurrences:
        start = max(0, position - radius)
        end = min(len(body), position + radius)
        window = body[start:end].strip()
        definition_score = len(_EXA_DEFINITION_CONTEXT.findall(window))
        candidate = (definition_score, -position, window)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2] if best is not None else ""


# --- value objects -------------------------------------------------------------


@dataclass(frozen=True)
class LLMResult:
    """Result of a single LLM completion."""

    text: str
    model: str
    finish_reason: str | None = "stop"
    usage: JsonDict = field(default_factory=dict)
    diagnostics: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class LLMStreamChunk:
    """One typed chunk from a streaming text completion."""

    delta: str = ""
    done: bool = False
    model: str | None = None
    provider: str | None = None
    usage: JsonDict = field(default_factory=dict)
    finish_reason: str | None = None
    diagnostics: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    """A normalized web search result."""

    title: str
    url: str
    snippet: str = ""
    source: str = ""
    rank: int = 0
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True)
class SearchCallResult:
    """Immutable query-local search result and provider outcome snapshot."""

    hits: tuple[SearchHit, ...]
    status: JsonDict


@dataclass(frozen=True)
class Document:
    """Fetched document content for indexing."""

    url: str
    title: str
    text: str
    content_type: str = "web"
    raw: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)


# --- protocols -----------------------------------------------------------------


@runtime_checkable
class LLMAdapter(Protocol):
    """Contract for text completion and embeddings."""

    name: str

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
        """Return a completion for the given chat messages."""
        ...

    def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        """Return embedding vectors for the given texts."""
        ...

    def estimate_cost(
        self,
        *,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> JsonDict:
        """Return a secret-free best-effort price estimate for a completion."""
        ...


@runtime_checkable
class StreamingLLMAdapter(Protocol):
    """Optional streaming capability detected at runtime for legacy adapters."""

    name: str

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
        """Yield assistant deltas followed by an optional terminal chunk."""
        ...

@runtime_checkable
class SearchToolAdapter(Protocol):
    """Contract for live web search and page fetch."""

    name: str

    def search(self, query: str, *, count: int = 5, options: SearchOptions | None = None) -> list[SearchHit]:
        """Return search hits for a query."""
        ...

    def fetch(self, url: str) -> Document:
        """Fetch and normalize a page into a Document."""
        ...

    def estimate_cost(self, *, searches: int = 0, results_per_search: int = 10) -> JsonDict:
        """Return a secret-free best-effort price estimate for search calls."""
        ...


# --- concrete implementations --------------------------------------------------


class OpenAICompatLLMAdapter:
    """LLMAdapter backed by an OpenAI-compatible endpoint (BYOK)."""

    name = "openai-compat"

    def __init__(
        self,
        client: LLMClient,
        *,
        embed_model: str = "",
        provider_type: str = "openai",
        provider_id: str = "",
    ) -> None:
        self._client = client
        self._embed_model = embed_model or client.model
        self._provider_type = provider_type or "openai"
        self._provider_id = provider_id or self._provider_type
        self.model = client.model

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
        """Complete via the wrapped OpenAI-compatible chat client.

        Thinking and output parameters are mapped only through the selected
        provider endpoint's declared capability contract.
        """

        selected_model = model or self._client.model
        call_options = resolve_llm_call_options(
            provider_type=self._provider_type,
            provider_id=self._provider_id,
            base_url=self._client.base_url,
            model=selected_model,
            thinking_budget=thinking_budget,
            max_visible_tokens=max_tokens,
            response_format=response_format,
        )
        result = self._client.chat_result(
            list(messages),
            temperature=temperature,
            request_fields=call_options.request_fields,
        )
        return LLMResult(
            text=result.text,
            model=result.model or selected_model,
            finish_reason=result.finish_reason,
            usage=result.usage,
            diagnostics={
                "classification": classify_llm_terminal(
                    text=result.text,
                    finish_reason=result.finish_reason,
                    usage=result.usage,
                    malformed=result.malformed_terminal,
                ),
                "call_options": call_options.to_dict(),
                "visible_chars": len(result.text),
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
        """Stream an OpenAI-compatible chat completion as typed chunks."""

        self._client.require_ready()
        selected_model = model or self._client.model
        call_options = resolve_llm_call_options(
            provider_type=self._provider_type,
            provider_id=self._provider_id,
            base_url=self._client.base_url,
            model=selected_model,
            thinking_budget=thinking_budget,
            max_visible_tokens=max_tokens,
            response_format=response_format,
        )
        payload: JsonDict = {
            "model": selected_model,
            "messages": list(messages),
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        payload.update(call_options.request_fields)
        request = urllib.request.Request(
            join_url(self._client.base_url, "/chat/completions"),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Authorization": f"Bearer {self._client.api_key}",
            },
            method="POST",
        )
        budget = current_provider_call_budget()
        timeout_s = 60.0
        if budget is not None:
            budget.check("openai_stream_connect")
            timeout_s = budget.call_timeout_s(timeout_s)
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                terminal_emitted = False
                visible_chars = 0
                final_finish_reason: str | None = None
                final_usage: JsonDict = {}
                malformed_chunks = 0
                for raw_line in response:
                    if budget is not None:
                        budget.check("openai_stream_read")
                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line or line.startswith(":") or not line.startswith("data:"):
                        continue
                    data_text = line[5:].strip()
                    if data_text == "[DONE]":
                        if not terminal_emitted:
                            yield LLMStreamChunk(
                                done=True,
                                model=selected_model,
                                provider=call_options.provider_identity,
                                usage=final_usage,
                                finish_reason=final_finish_reason,
                                diagnostics=_terminal_diagnostics(
                                    text_length=visible_chars,
                                    finish_reason=final_finish_reason,
                                    usage=final_usage,
                                    call_options=call_options.to_dict(),
                                    malformed=malformed_chunks > 0,
                                ),
                            )
                            terminal_emitted = True
                        break
                    try:
                        data = json.loads(data_text)
                    except json.JSONDecodeError:
                        malformed_chunks += 1
                        continue
                    choices = data.get("choices") if isinstance(data, dict) else None
                    choice = choices[0] if isinstance(choices, list) and choices else {}
                    delta_obj = choice.get("delta") if isinstance(choice, dict) else {}
                    delta = _stream_content_text(
                        (delta_obj or {}).get("content") if isinstance(delta_obj, dict) else None
                    )
                    if not delta and isinstance(choice, dict):
                        delta = _stream_content_text(choice.get("text"))
                    finish_reason = choice.get("finish_reason") if isinstance(choice, dict) else None
                    usage = dict(data.get("usage") or {}) if isinstance(data, dict) else {}
                    visible_chars += len(delta)
                    if finish_reason is not None:
                        final_finish_reason = str(finish_reason)
                    if usage:
                        final_usage.update(usage)
                    is_terminal = bool(finish_reason)
                    yield LLMStreamChunk(
                        delta=delta,
                        done=is_terminal,
                        model=str(data.get("model") or selected_model),
                        provider=call_options.provider_identity,
                        usage=usage,
                        finish_reason=str(finish_reason) if finish_reason is not None else None,
                        diagnostics=(
                            _terminal_diagnostics(
                                text_length=visible_chars,
                                finish_reason=final_finish_reason,
                                usage=final_usage,
                                call_options=call_options.to_dict(),
                                malformed=malformed_chunks > 0,
                            )
                            if is_terminal
                            else {}
                        ),
                    )
                    terminal_emitted = terminal_emitted or is_terminal
                if not terminal_emitted:
                    yield LLMStreamChunk(
                        done=True,
                        model=selected_model,
                        provider=call_options.provider_identity,
                        usage=final_usage,
                        finish_reason=final_finish_reason,
                        diagnostics=_terminal_diagnostics(
                            text_length=visible_chars,
                            finish_reason=final_finish_reason,
                            usage=final_usage,
                            call_options=call_options.to_dict(),
                            malformed=malformed_chunks > 0,
                        ),
                    )
                if budget is not None:
                    budget.check("openai_stream_read")
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            if budget is not None:
                budget.check("openai_stream_read")
            raise BaselineError(
                ErrorCode.UPSTREAM_ERROR,
                redact_text(f"streaming completion failed: {error}"),
            ) from error

    def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]]:
        """Embed via the OpenAI-compatible ``/embeddings`` endpoint."""

        self._client.require_ready()
        payload = {"model": model or self._embed_model, "input": list(texts)}
        request = urllib.request.Request(
            join_url(self._client.base_url, "/embeddings"),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {self._client.api_key}",
            },
            method="POST",
        )
        data = request_json_with_retries(
            self._client.transport,
            request,
            retry_policy=self._client.retry_policy,
            sleeper=self._client.sleeper,
        )
        return _parse_embeddings(data, expected=len(texts))

    def estimate_cost(
        self,
        *,
        model: str | None = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> JsonDict:
        """Estimate completion cost without making an upstream request."""

        low, high, basis, disclaimer = estimate_llm_cost(
            model_id=model or self._client.model,
            provider_type=self._provider_type,
            input_tokens=max(0, int(input_tokens)),
            output_tokens=max(0, int(output_tokens)),
        )
        return {"usd_low": low, "usd_high": high, "basis": basis, "disclaimer": disclaimer}


def classify_llm_terminal(
    *,
    text: str,
    finish_reason: str | None,
    usage: JsonDict | None,
    malformed: bool = False,
) -> str:
    """Return the stable T52 terminal classification without raw response data."""

    visible = bool(str(text or "").strip())
    finish = str(finish_reason or "").strip().lower()
    observed_usage = bool(usage)
    if visible:
        return "completed" if observed_usage else "usage_unavailable"
    if finish in {"length", "max_tokens", "max_output_tokens"}:
        return "length_without_visible_output"
    if finish in {"content_filter", "content_filtered", "safety"}:
        return "content_filtered"
    if finish in {"stop", "end_turn"}:
        return "provider_stop_empty"
    if observed_usage:
        return "usage_only"
    if malformed:
        return "malformed_terminal"
    return "empty_stream"


def _terminal_diagnostics(
    *,
    text_length: int,
    finish_reason: str | None,
    usage: JsonDict,
    call_options: JsonDict,
    malformed: bool,
) -> JsonDict:
    return {
        "classification": classify_llm_terminal(
            text="x" * max(0, int(text_length)),
            finish_reason=finish_reason,
            usage=usage,
            malformed=malformed,
        ),
        "call_options": call_options,
        "visible_chars": max(0, int(text_length)),
    }


def _stream_content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        return ""
    parts: list[str] = []
    for item in value:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict):
            text = item.get("text") or item.get("content") or item.get("output_text")
            if isinstance(text, str):
                parts.append(text)
            elif isinstance(text, dict) and isinstance(text.get("value"), str):
                parts.append(text["value"])
    return "".join(parts)


class BraveFirecrawlSearchAdapter:
    """SearchToolAdapter backed by Brave Search + Firecrawl scrape (BYOK)."""

    name = "brave-firecrawl"

    def __init__(self, client: SearchClient) -> None:
        self._client = client

    def search(self, query: str, *, count: int = 5, options: SearchOptions | None = None) -> list[SearchHit]:
        """Search the web and return normalized hits."""

        raw = self._client.search(query, count=count, options=options)
        return [
            SearchHit(
                title=str(item.get("title") or item.get("url") or ""),
                url=str(item.get("url") or ""),
                snippet=str(item.get("description") or ""),
                source=str(item.get("source") or ""),
                rank=index,
                metadata={"fallback_tier": "brave", "search_provider": "brave"},
            )
            for index, item in enumerate(raw)
            if item.get("url")
        ]

    def fetch(self, url: str) -> Document:
        """Fetch a page; fall back to empty text on scrape failure."""

        try:
            scraped = self._client.scrape(url)
        except Exception as error:  # noqa: BLE001 - normalized to safe text
            raise BaselineError(
                ErrorCode.UPSTREAM_ERROR,
                redact_text(f"fetch failed for {url}: {error}"),
            ) from error
        return Document(
            url=url,
            title=url,
            text=str(scraped.get("markdown") or ""),
            raw=scraped.get("raw", {}),
            metadata={"fetch_provider": "firecrawl"},
        )

    def estimate_cost(self, *, searches: int = 0, results_per_search: int = 10) -> JsonDict:
        """Estimate Brave Search cost without making an upstream request."""

        low, high, credits_low, credits_high, basis, disclaimer = estimate_search_cost(
            search_provider="brave",
            searches=max(0, int(searches)),
            results_per_search=max(1, int(results_per_search)),
        )
        return {
            "usd_low": low,
            "usd_high": high,
            "credits_low": credits_low,
            "credits_high": credits_high,
            "basis": basis,
            "disclaimer": disclaimer,
        }


class ResilientSearchAdapter:
    """SearchToolAdapter with provider -> cache -> indexed-external fallback."""

    name = "resilient-search"
    supports_search_usage_policy = True
    supports_relevance_aware_fallback = True
    supports_focused_fetch = True

    def __init__(
        self,
        client: SearchClient,
        *,
        store: Any,
        retriever: Any,
        ttl_s: float = 3600.0,
    ) -> None:
        self._client = client
        self._store = store
        self._retriever = retriever
        self._ttl_s = ttl_s
        self._last_search_status: JsonDict = {}

    def search_cache_key(
        self,
        query: str,
        *,
        count: int = 5,
        options: SearchOptions | None = None,
        resolved_usage: JsonDict | None = None,
    ) -> str:
        """Return a stable cache key for search results."""

        settings = getattr(self._client, "settings", None)
        option_payload = options.to_dict() if options else None
        if option_payload is not None:
            for field_name in ("country", "search_lang", "ui_language"):
                option_payload.pop(field_name, None)
            if not any(
                (
                    option_payload.get("include_domains"),
                    option_payload.get("exclude_domains"),
                    option_payload.get("freshness"),
                    option_payload.get("extra_subqueries"),
                )
            ):
                option_payload = None
        effective_locale = (options or SearchOptions()).effective_locale(settings)
        payload = {
            "version": 4,
            "endpoint": "web-search",
            "query": query,
            "count": max(1, min(count, 100)),
            "effective_locale": effective_locale,
            "sources": ["web"],
            "options": option_payload,
            "resolved_search_chain": _resolved_provider_chain(
                resolved_usage,
                "search_chain",
                legacy=("brave", "firecrawl"),
            ),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return "search:" + hashlib.sha256(encoded).hexdigest()

    def fetch_cache_key(
        self,
        url: str,
        *,
        relevance_query: str | None = None,
        resolved_usage: JsonDict | None = None,
    ) -> str:
        """Return a stable cache key for fetched page text."""

        encoded = json.dumps(
            {
                "version": 3,
                "url": _canonical_cache_url(url),
                "focus": str(relevance_query or "").strip(),
                "resolved_extract_chain": _resolved_provider_chain(
                    resolved_usage,
                    "extract_chain",
                    legacy=("firecrawl",),
                ),
            },
            sort_keys=True,
        ).encode("utf-8")
        return "fetch:" + hashlib.sha256(encoded).hexdigest()

    def estimate_cost(self, *, searches: int = 0, results_per_search: int = 10) -> JsonDict:
        """Estimate primary Brave and fallback Firecrawl search exposure."""

        settings = getattr(self._client, "settings", None)
        provider = (
            "mixed_provider_unknown"
            if bool(getattr(settings, "exa_api_key", "") or getattr(settings, "tavily_api_key", ""))
            else "brave_firecrawl"
        )
        low, high, credits_low, credits_high, basis, disclaimer = estimate_search_cost(
            search_provider=provider,
            searches=max(0, int(searches)),
            results_per_search=max(1, int(results_per_search)),
        )
        return {
            "usd_low": low,
            "usd_high": high,
            "credits_low": credits_low,
            "credits_high": credits_high,
            "basis": basis,
            "disclaimer": disclaimer,
        }

    def search(
        self,
        query: str,
        *,
        count: int = 5,
        options: SearchOptions | None = None,
        resolved_usage: JsonDict | None = None,
    ) -> list[SearchHit]:
        """Search the external provenance lane with ordered provider/cache/index fallback."""

        result = self.search_with_outcome(
            query,
            count=count,
            options=options,
            resolved_usage=resolved_usage,
        )
        return list(result.hits)

    def search_with_outcome(
        self,
        query: str,
        *,
        count: int = 5,
        options: SearchOptions | None = None,
        recovery: bool = False,
        excluded_provenance_roots: Sequence[str] = (),
        relevance_query: str | None = None,
        resolved_usage: JsonDict | None = None,
        provider_circuit: set[str] | None = None,
    ) -> SearchCallResult:
        """Run the legacy synchronous provider/cache/index fallback path."""

        network = self.search_network(
            query,
            count=count,
            options=options,
            recovery=recovery,
            excluded_provenance_roots=excluded_provenance_roots,
            relevance_query=relevance_query,
            resolved_usage=resolved_usage,
            provider_circuit=provider_circuit,
        )
        if network.hits:
            self.persist_search_cache(
                query,
                count=count,
                options=options,
                result=network,
                resolved_usage=resolved_usage,
            )
            self._remember_search_result(network)
            return network

        if not recovery:
            cached = self.read_search_cache(
                query,
                count=count,
                options=options,
                relevance_query=relevance_query,
                upstream_status=network.status,
                resolved_usage=resolved_usage,
            )
            if cached is not None:
                self._remember_search_result(cached)
                return cached

        indexed = self.read_indexed_external(
            query,
            count=count,
            options=options,
            recovery=recovery,
            excluded_provenance_roots=excluded_provenance_roots,
            relevance_query=relevance_query,
            upstream_status=network.status,
            resolved_usage=resolved_usage,
        )
        self._remember_search_result(indexed)
        return indexed

    def search_network(
        self,
        query: str,
        *,
        count: int = 5,
        options: SearchOptions | None = None,
        recovery: bool = False,
        excluded_provenance_roots: Sequence[str] = (),
        relevance_query: str | None = None,
        resolved_usage: JsonDict | None = None,
        provider_circuit: set[str] | None = None,
    ) -> SearchCallResult:
        """Run only live provider calls; never touch Store or shared status."""

        attempted: list[str] = []
        failed: list[str] = []
        empty: list[str] = []
        irrelevant: list[str] = []
        skipped: list[str] = []
        failure_reasons: dict[str, str] = {}
        deferred_unknown: tuple[str, list[SearchHit]] | None = None
        circuit = provider_circuit if provider_circuit is not None else set()
        calls = {
            "brave": lambda: self._client.search(query, count=count, options=options),
            "tavily": lambda: self._client.tavily_search(query, count=count, options=options),
            "exa": lambda: self._client.exa_search(query, count=count, options=options),
            "firecrawl": lambda: self._client.firecrawl_search(query, count=count, options=options),
            "firecrawl_search": lambda: self._client.firecrawl_search(query, count=count, options=options),
        }
        chain = _resolved_provider_chain(
            resolved_usage,
            "search_chain",
            legacy=("brave", "firecrawl_search"),
        )
        for provider_index, tier in enumerate(chain):
            provider = _normalized_provider_id(tier)
            if provider in circuit:
                skipped.append(provider)
                continue
            call = calls.get(tier)
            if call is None:
                failed.append(tier)
                failure_reasons[provider] = "unsupported_provider"
                continue
            attempted.append(tier)
            parent_budget = current_provider_call_budget()
            call_budget = parent_budget
            remaining_provider_count = sum(
                1
                for next_tier in chain[provider_index:]
                if _normalized_provider_id(next_tier) not in circuit
                and calls.get(next_tier) is not None
            )
            if parent_budget is not None and remaining_provider_count > 1:
                call_budget = parent_budget.with_timeout_cap(
                    max(1.0, parent_budget.remaining_s() / remaining_provider_count)
                )
            try:
                with use_provider_call_budget(call_budget):
                    raw = call()
            except BaselineError as error:
                if error.code is ErrorCode.RUN_CANCELLED:
                    raise
                if error.code is ErrorCode.RUN_TIMEOUT:
                    if (
                        call_budget is parent_budget
                        or parent_budget is None
                        or parent_budget.remaining_s() <= 0
                    ):
                        raise
                    failed.append(tier)
                    failure_reasons[provider] = "timeout"
                    continue
                failed.append(tier)
                failure_reasons[provider] = "upstream_error"
                continue
            except Exception as error:  # noqa: BLE001 - normalized below
                failed.append(tier)
                reason = _provider_failure_reason(error)
                failure_reasons[provider] = reason
                if reason in {"auth_failed", "payment_required"}:
                    circuit.add(provider)
                continue
            hits = self._hits_from_raw(raw, tier=tier)
            if relevance_query:
                hits = self._relevance_hits(
                    hits,
                    excluded_provenance_roots=excluded_provenance_roots,
                    relevance_query=relevance_query,
                )
                if raw and not hits:
                    irrelevant.append(tier)
                    continue
                if (
                    recovery
                    and hits
                    and not self._recovery_tier_is_fetch_ready(
                        hits,
                        relevance_query=relevance_query,
                    )
                ):
                    if deferred_unknown is None:
                        deferred_unknown = (tier, hits)
                    continue
            if hits:
                locale = _search_locale_metadata(options, getattr(self._client, "settings", None), tier=tier)
                return self._build_search_result(
                    hits,
                    tier=tier,
                    attempted=attempted,
                    failed=failed,
                    empty=empty,
                    irrelevant=irrelevant,
                    locale=locale,
                    search_options=_search_option_metadata(options, tier=tier),
                    producer_tier=tier,
                    resolved_usage=resolved_usage,
                    skipped=skipped,
                    failure_reasons=failure_reasons,
                )
            empty.append(tier)

        if deferred_unknown is not None:
            tier, hits = deferred_unknown
            return self._build_search_result(
                hits,
                tier=tier,
                attempted=attempted,
                failed=failed,
                empty=empty,
                irrelevant=irrelevant,
                locale=_search_locale_metadata(
                    options,
                    getattr(self._client, "settings", None),
                    tier=tier,
                ),
                search_options=_search_option_metadata(options, tier=tier),
                producer_tier=tier,
                resolved_usage=resolved_usage,
                skipped=skipped,
                failure_reasons=failure_reasons,
            )

        return self._build_search_result(
            [],
            tier="none",
            attempted=attempted,
            failed=failed,
            empty=empty,
            irrelevant=irrelevant,
            locale=_search_locale_metadata(
                options,
                getattr(self._client, "settings", None),
                tier="none",
            ),
            search_options=_search_option_metadata(options, tier="none"),
            producer_tier="none",
            resolved_usage=resolved_usage,
            skipped=skipped,
            failure_reasons=failure_reasons,
        )

    def read_search_cache(
        self,
        query: str,
        *,
        count: int = 5,
        options: SearchOptions | None = None,
        relevance_query: str | None = None,
        upstream_status: JsonDict | None = None,
        resolved_usage: JsonDict | None = None,
    ) -> SearchCallResult | None:
        """Read a fresh search-cache snapshot without changing shared status."""

        cached = self._store.cache_get(
            self.search_cache_key(
                query,
                count=count,
                options=options,
                resolved_usage=resolved_usage,
            )
        )
        if not cached:
            return None
        hits = self._hits_from_cache(cached)
        if relevance_query:
            hits = self._relevance_hits(
                hits,
                excluded_provenance_roots=(),
                relevance_query=relevance_query,
            )
        if not hits:
            return None
        upstream = dict(upstream_status or {})
        producer_tier = str(cached.get("producer_tier") or "cache")
        return self._build_search_result(
            hits,
            tier="cache",
            attempted=list(upstream.get("providers_attempted") or []),
            failed=list(upstream.get("failed_providers") or []),
            empty=list(upstream.get("empty_providers") or []),
            irrelevant=list(upstream.get("irrelevant_providers") or []),
            locale=_search_locale_metadata(
                options,
                getattr(self._client, "settings", None),
                tier=producer_tier,
                producer_locale=(
                    dict(cached.get("locale"))
                    if isinstance(cached.get("locale"), dict)
                    else None
                ),
            ),
            search_options=_search_option_metadata(
                options,
                tier=producer_tier,
                producer_metadata=(
                    dict(cached.get("search_options"))
                    if isinstance(cached.get("search_options"), dict)
                    else None
                ),
            ),
            producer_tier=producer_tier,
            resolved_usage=resolved_usage,
            skipped=list(upstream.get("skipped_providers") or []),
            failure_reasons=dict(upstream.get("provider_failure_reasons") or {}),
        )

    def persist_search_cache(
        self,
        query: str,
        *,
        count: int,
        options: SearchOptions | None,
        result: SearchCallResult,
        resolved_usage: JsonDict | None = None,
    ) -> bool:
        """Explicitly persist a live immutable result on the caller thread."""

        producer_tier = str(result.status.get("producer_tier") or "")
        if producer_tier not in {"brave", "tavily", "exa", "firecrawl", "firecrawl_search"} or not result.hits:
            return False
        locale = result.status.get("locale")
        search_options = result.status.get("search_options")
        self._write_search_cache(
            query,
            count=count,
            options=options,
            hits=list(result.hits),
            producer_tier=producer_tier,
            locale=dict(locale) if isinstance(locale, dict) else {},
            search_options=(
                dict(search_options) if isinstance(search_options, dict) else {}
            ),
            resolved_usage=resolved_usage,
        )
        return True

    def read_indexed_external(
        self,
        query: str,
        *,
        count: int = 5,
        options: SearchOptions | None = None,
        recovery: bool = False,
        excluded_provenance_roots: Sequence[str] = (),
        relevance_query: str | None = None,
        upstream_status: JsonDict | None = None,
        resolved_usage: JsonDict | None = None,
    ) -> SearchCallResult:
        """Read the indexed-external fallback on the caller/Store thread."""

        hits = self._local_hits(query, count=count, resolved_usage=resolved_usage)
        if relevance_query:
            hits = self._relevance_hits(
                hits,
                excluded_provenance_roots=excluded_provenance_roots,
                relevance_query=relevance_query,
            )
        upstream = dict(upstream_status or {})
        tier = "indexed_external" if hits else "none"
        return self._build_search_result(
            hits,
            tier=tier,
            attempted=list(upstream.get("providers_attempted") or []),
            failed=list(upstream.get("failed_providers") or []),
            empty=list(upstream.get("empty_providers") or []),
            irrelevant=list(upstream.get("irrelevant_providers") or []),
            locale=_search_locale_metadata(
                options,
                getattr(self._client, "settings", None),
                tier=tier,
            ),
            search_options=_search_option_metadata(options, tier=tier),
            producer_tier=tier,
            resolved_usage=resolved_usage,
            skipped=list(upstream.get("skipped_providers") or []),
            failure_reasons=dict(upstream.get("provider_failure_reasons") or {}),
        )

    @staticmethod
    def _build_search_result(
        hits: list[SearchHit],
        *,
        tier: str,
        attempted: list[str],
        failed: list[str],
        empty: list[str],
        irrelevant: list[str],
        locale: JsonDict,
        search_options: JsonDict,
        producer_tier: str,
        resolved_usage: JsonDict | None = None,
        skipped: list[str] | None = None,
        failure_reasons: JsonDict | None = None,
    ) -> SearchCallResult:
        status = {
            "fallback_tier": tier,
            "live_providers_down": bool(failed),
            "providers_attempted": list(attempted),
            "failed_providers": list(failed),
            "empty_providers": list(empty),
            "irrelevant_providers": list(irrelevant),
            "skipped_providers": list(skipped or []),
            "provider_failure_reasons": dict(failure_reasons or {}),
            "producer_tier": producer_tier,
            "selected_provider": (
                "firecrawl" if producer_tier == "firecrawl_search" else producer_tier
            ) if producer_tier not in {"none", "cache", "indexed_external"} else None,
            "resolved_search_chain": _resolved_provider_chain(
                resolved_usage,
                "search_chain",
                legacy=("brave", "firecrawl"),
            ),
            "resolved_extract_chain": _resolved_provider_chain(
                resolved_usage,
                "extract_chain",
                legacy=("firecrawl",),
            ),
            "policy_reason_codes": list((resolved_usage or {}).get("reason_codes") or []),
            "locale": locale,
            "search_options": search_options,
        }
        return SearchCallResult(hits=tuple(hits), status=dict(status))

    def _remember_search_result(self, result: SearchCallResult) -> None:
        """Maintain the legacy synchronous status API outside pure network calls."""

        self._last_search_status = dict(result.status)

    @staticmethod
    def _relevance_hits(
        hits: list[SearchHit],
        *,
        excluded_provenance_roots: Sequence[str],
        relevance_query: str | None,
    ) -> list[SearchHit]:
        """Drop only high-confidence rejected hits before choosing a provider tier."""

        excluded = {str(value) for value in excluded_provenance_roots if str(value)}
        remaining = [
            hit
            for hit in hits
            if str(hit.metadata.get("provenance_root") or hit.url) not in excluded
        ]
        if not relevance_query or not remaining:
            return remaining

        # Lazy imports keep provider transport independent from retrieval at module load.
        # ``unknown`` remains fetchable: only a deterministic rejection may advance the
        # provider chain, so scorer uncertainty cannot silently discard useful evidence.
        from .evidence_eligibility import parse_research_intent
        from .external_candidates import candidate_from_search_hit

        intent = parse_research_intent(relevance_query)
        return [
            hit
            for hit in remaining
            if candidate_from_search_hit(
                hit,
                plan_id="provider-chain-prefilter",
                query=relevance_query,
                intent=intent,
            ).prefetch_decision
            != "rejected"
        ]

    @staticmethod
    def _recovery_tier_is_fetch_ready(
        hits: list[SearchHit],
        *,
        relevance_query: str,
    ) -> bool:
        """Prefer a stronger recovery tier while retaining deferred unknown hits."""

        from .evidence_eligibility import parse_research_intent
        from .external_candidates import candidate_from_search_hit

        intent = parse_research_intent(relevance_query)
        for hit in hits:
            candidate = candidate_from_search_hit(
                hit,
                plan_id="provider-chain-recovery",
                query=relevance_query,
                intent=intent,
            )
            if candidate.prefetch_decision == "candidate":
                return True
            if (
                candidate.prefetch_decision == "unknown"
                and "source_attribution_identity_matched" in candidate.reason_codes
                and "source_attribution_identity_mismatch" not in candidate.reason_codes
            ):
                return True
        return False

    def last_search_status(self) -> JsonDict:
        """Return the most recent query-level provider/cache fallback status."""

        return dict(self._last_search_status)

    def fetch(
        self,
        url: str,
        *,
        relevance_query: str | None = None,
        resolved_usage: JsonDict | None = None,
        provider_circuit: set[str] | None = None,
    ) -> Document:
        """Run the legacy synchronous cache -> network -> cache path."""

        cached = self.read_fetch_cache(
            url,
            relevance_query=relevance_query,
            resolved_usage=resolved_usage,
        )
        if cached is not None:
            return cached
        document = self.fetch_network(
            url,
            relevance_query=relevance_query,
            resolved_usage=resolved_usage,
            provider_circuit=provider_circuit,
        )
        self.persist_fetch_cache(
            document,
            relevance_query=relevance_query,
            resolved_usage=resolved_usage,
        )
        return document

    def read_fetch_cache(
        self,
        url: str,
        *,
        relevance_query: str | None = None,
        resolved_usage: JsonDict | None = None,
    ) -> Document | None:
        """Read one fetch-cache entry on the caller/Store thread."""

        cached = self._store.cache_get(
            self.fetch_cache_key(
                url,
                relevance_query=relevance_query,
                resolved_usage=resolved_usage,
            )
        )
        if not cached:
            return None
        metadata = dict(cached.get("metadata") or {})
        if cached.get("producer_provider"):
            metadata.setdefault("producer_provider", str(cached["producer_provider"]))
        if cached.get("content_basis"):
            metadata.setdefault("content_basis", str(cached["content_basis"]))
        return Document(
            url=str(cached.get("url") or url),
            title=str(cached.get("title") or url),
            text=str(cached.get("text") or ""),
            content_type=str(cached.get("content_type") or "web"),
            metadata=metadata,
        )

    def fetch_network(
        self,
        url: str,
        *,
        relevance_query: str | None = None,
        resolved_usage: JsonDict | None = None,
        provider_circuit: set[str] | None = None,
    ) -> Document:
        """Fetch one live page without reading or writing Store state."""

        calls = {
            "tavily": lambda: self._client.tavily_extract(url),
            "exa": lambda: self._exa_contents(url, relevance_query=relevance_query),
            "firecrawl": lambda: self._client.scrape(url),
        }
        failed: list[str] = []
        skipped: list[str] = []
        failure_reasons: dict[str, str] = {}
        circuit = provider_circuit if provider_circuit is not None else set()
        extract_chain = _resolved_provider_chain(
            resolved_usage,
            "extract_chain",
            legacy=("firecrawl",),
        )
        for provider_index, provider in enumerate(extract_chain):
            normalized = _normalized_provider_id(provider)
            if normalized in circuit:
                skipped.append(normalized)
                continue
            call = calls.get(provider)
            if call is None:
                failed.append(provider)
                failure_reasons[normalized] = "unsupported_provider"
                continue
            parent_budget = current_provider_call_budget()
            call_budget = parent_budget
            has_fallback = any(
                _normalized_provider_id(next_provider) not in circuit
                and calls.get(next_provider) is not None
                for next_provider in extract_chain[provider_index + 1 :]
            )
            if parent_budget is not None and has_fallback:
                usable_s = parent_budget.remaining_s()
                fallback_reserve_s = min(5.0, usable_s / 2.0)
                call_budget = parent_budget.with_timeout_cap(
                    max(0.0, usable_s - fallback_reserve_s)
                )
            try:
                with use_provider_call_budget(call_budget):
                    scraped = call()
            except BaselineError as error:
                if error.code is ErrorCode.RUN_CANCELLED:
                    raise
                if error.code is ErrorCode.RUN_TIMEOUT:
                    if (
                        call_budget is parent_budget
                        or parent_budget is None
                        or parent_budget.remaining_s() <= 0
                    ):
                        raise
                    failed.append(provider)
                    failure_reasons[normalized] = "timeout"
                    continue
                failed.append(provider)
                failure_reasons[normalized] = "upstream_error"
                continue
            except Exception as error:  # noqa: BLE001 - fallback continues with secret-free diagnostics
                failed.append(provider)
                reason = _provider_failure_reason(error)
                failure_reasons[normalized] = reason
                if reason in {"auth_failed", "payment_required"}:
                    circuit.add(normalized)
                continue
            raw = scraped.get("raw", {})
            raw = dict(raw) if isinstance(raw, dict) else {}
            canonical_url = str(scraped.get("canonical_url") or "").strip() or _canonical_url_from_scrape(raw)
            text = str(scraped.get("markdown") or "")
            if not text.strip():
                failed.append(provider)
                continue
            return Document(
                url=url,
                title=str(raw.get("title") or url),
                text=text,
                raw=raw,
                metadata={
                    "fallback_tier": "live_fetch",
                    "producer_provider": provider,
                    "fetch_provider": provider,
                    "content_basis": "fetched",
                    "attempted_extract_providers": [*failed, provider],
                    "skipped_extract_providers": skipped,
                    "provider_failure_reasons": failure_reasons,
                    **({"canonical_url": canonical_url} if canonical_url else {}),
                },
            )
        raise BaselineError(
            ErrorCode.UPSTREAM_ERROR,
            redact_text(
                f"fetch failed for {url}; attempted={','.join(failed) or 'none'}; "
                f"skipped={','.join(skipped) or 'none'}"
            ),
        )

    def persist_fetch_cache(
        self,
        document: Document,
        *,
        relevance_query: str | None = None,
        resolved_usage: JsonDict | None = None,
    ) -> None:
        """Explicitly persist fetched content on the caller/Store thread."""

        self._write_fetch_cache(
            document,
            relevance_query=relevance_query,
            resolved_usage=resolved_usage,
        )

    def _exa_contents(self, url: str, *, relevance_query: str | None) -> JsonDict:
        """Request focused highlights plus a bounded long-document fallback when needed."""

        call = self._client.exa_contents
        try:
            parameters = tuple(inspect.signature(call).parameters.values())
            supports_kwargs = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
            )
            supports_focus = any(
                parameter.name == "highlights_query"
                for parameter in parameters
            ) or supports_kwargs
            supports_max_characters = any(
                parameter.name == "max_characters" for parameter in parameters
            ) or supports_kwargs
        except (TypeError, ValueError):
            supports_focus = False
            supports_max_characters = False
        if relevance_query and supports_focus:
            kwargs: JsonDict = {"highlights_query": relevance_query}
            extended_focus = _EXA_EXTENDED_TEXT_FOCUS.search(relevance_query)
            if supports_max_characters and extended_focus:
                # Long standards and protocol documents can place a requested numeric
                # definition beyond Exa's ordinary bounded text prefix. Highlights remain
                # the preferred extractive path; this same-call body is only its fallback.
                kwargs["max_characters"] = _EXA_EXTENDED_TEXT_MAX_CHARACTERS
            result = call(url, **kwargs)
            if extended_focus:
                focused_body = _focused_exa_numeric_body_excerpt(
                    result,
                    relevance_query=relevance_query,
                )
                markdown = str(result.get("markdown") or "").strip()
                if focused_body and focused_body not in markdown:
                    return {
                        **result,
                        "markdown": "\n\n".join(
                            value for value in (markdown, focused_body) if value
                        ),
                    }
            return result
        return call(url)

    def _hits_from_raw(self, raw: Sequence[JsonDict], *, tier: str) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for index, item in enumerate(raw):
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            nested_metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
            search_provider = str(
                nested_metadata.get("search_provider")
                or item.get("search_provider")
                or ("firecrawl" if tier == "firecrawl_search" else tier)
            )
            hits.append(
                SearchHit(
                    title=str(item.get("title") or url),
                    url=url,
                    snippet=str(item.get("description") or item.get("snippet") or ""),
                    source=str(item.get("source") or urllib.parse.urlparse(url).netloc),
                    rank=index,
                    metadata={
                        **nested_metadata,
                        "fallback_tier": tier,
                        "search_provider": search_provider,
                        "content_basis": str(nested_metadata.get("content_basis") or "snippet"),
                        **_temporal_metadata(item),
                    },
                )
            )
        return hits

    def _hits_from_cache(self, cached: JsonDict) -> list[SearchHit]:
        raw_hits = cached.get("hits") if isinstance(cached, dict) else None
        if not isinstance(raw_hits, list):
            return []
        hits = self._hits_from_raw(raw_hits, tier="cache")
        return [
            SearchHit(
                title=hit.title,
                url=hit.url,
                snippet=hit.snippet,
                source=hit.source,
                rank=hit.rank,
                metadata={**hit.metadata, "fallback_tier": "cache"},
            )
            for hit in hits
        ]

    def _local_hits(
        self,
        query: str,
        *,
        count: int,
        resolved_usage: JsonDict | None = None,
    ) -> list[SearchHit]:
        try:
            evidence = self._retriever.retrieve(query, top_k=count)
        except Exception:
            return []
        hits: list[SearchHit] = []
        for index, item in enumerate(evidence):
            url = str(getattr(item, "url", "") or "")
            if not url.startswith(("http://", "https://")):
                continue
            temporal_metadata = {
                key: value
                for key in ("published_at", "updated_at", "effective_at", "retrieved_at")
                if (value := getattr(item, key, None)) is not None
            }
            document = Document(
                url=url,
                title=str(getattr(item, "title", "") or url),
                text=str(getattr(item, "text", "") or ""),
                content_type="web",
                metadata={
                    "fallback_tier": "indexed_external",
                    "producer_provider": "indexed_external",
                    "fetch_provider": "indexed_external",
                    "content_basis": "fetched",
                    **temporal_metadata,
                },
            )
            self._write_fetch_cache(document, resolved_usage=resolved_usage)
            hits.append(
                SearchHit(
                    title=document.title,
                    url=document.url,
                    snippet=document.text[:240],
                    source=urllib.parse.urlparse(document.url).netloc,
                    rank=index,
                    metadata={
                        "fallback_tier": "indexed_external",
                        "producer_provider": "indexed_external",
                        "search_provider": "indexed_external",
                        "content_basis": "snippet",
                        "document_id": getattr(item, "document_id", ""),
                        "chunk_id": getattr(item, "chunk_id", ""),
                        **temporal_metadata,
                    },
                )
            )
        return hits

    def _write_search_cache(
        self,
        query: str,
        *,
        count: int,
        options: SearchOptions | None,
        hits: list[SearchHit],
        producer_tier: str,
        locale: JsonDict,
        search_options: JsonDict,
        resolved_usage: JsonDict | None = None,
    ) -> None:
        self._store.cache_set(
            self.search_cache_key(
                query,
                count=count,
                options=options,
                resolved_usage=resolved_usage,
            ),
            {
                "version": 4,
                "producer_tier": producer_tier,
                "producer_provider": "firecrawl" if producer_tier == "firecrawl_search" else producer_tier,
                "locale": locale,
                "search_options": search_options,
                "hits": [_hit_to_cache(hit) for hit in hits],
                "resolved_search_chain": _resolved_provider_chain(
                    resolved_usage,
                    "search_chain",
                    legacy=("brave", "firecrawl"),
                ),
            },
            ttl_s=self._ttl_s,
        )

    def _write_fetch_cache(
        self,
        document: Document,
        *,
        relevance_query: str | None = None,
        resolved_usage: JsonDict | None = None,
    ) -> None:
        self._store.cache_set(
            self.fetch_cache_key(
                document.url,
                relevance_query=relevance_query,
                resolved_usage=resolved_usage,
            ),
            {
                "version": 3,
                "url": document.url,
                "title": document.title,
                "text": document.text,
                "content_type": document.content_type,
                "metadata": document.metadata,
                "producer_provider": str(
                    document.metadata.get("producer_provider")
                    or document.metadata.get("fetch_provider")
                    or document.metadata.get("fallback_tier")
                    or ""
                ),
                "content_basis": str(document.metadata.get("content_basis") or "fetched"),
                "resolved_extract_chain": _resolved_provider_chain(
                    resolved_usage,
                    "extract_chain",
                    legacy=("firecrawl",),
                ),
            },
            ttl_s=self._ttl_s,
        )


def _hit_to_cache(hit: SearchHit) -> JsonDict:
    return {
        "title": hit.title,
        "url": hit.url,
        "description": hit.snippet,
        "source": hit.source,
        "metadata": dict(hit.metadata),
    }


def _canonical_url_from_scrape(raw: JsonDict) -> str:
    """Extract a provider-declared document URL without treating it as trusted."""

    nested = raw.get("metadata")
    sources = (raw, nested if isinstance(nested, dict) else {})
    for source in sources:
        for key in ("canonical_url", "canonicalUrl", "source_url", "sourceURL", "url"):
            value = str(source.get(key) or "").strip()
            if value.startswith(("http://", "https://")):
                return value
    return ""


def _resolved_provider_chain(
    resolved_usage: JsonDict | None,
    field: str,
    *,
    legacy: Sequence[str],
) -> list[str]:
    """Return a sanitized frozen provider chain or the legacy compatibility chain."""

    allowed = {"brave", "tavily", "exa", "firecrawl", "firecrawl_search"}
    raw = (resolved_usage or {}).get(field)
    if not isinstance(raw, list):
        return list(legacy)
    chain = [str(provider).strip().lower() for provider in raw]
    return list(dict.fromkeys(provider for provider in chain if provider in allowed))


def _canonical_cache_url(value: object) -> str:
    """Return the same conservative document identity used by T49 candidates."""

    # Lazy import avoids the adapters -> retrieval -> adapters import cycle.
    from .external_candidates import canonicalize_external_url

    return canonicalize_external_url(value) or str(value or "").strip()


def _normalized_provider_id(value: object) -> str:
    provider = str(value or "").strip().lower()
    return "firecrawl" if provider == "firecrawl_search" else provider


def _provider_failure_reason(error: Exception) -> str:
    """Classify transport failures without persisting provider response bodies."""

    status_code = getattr(error, "status_code", None)
    if isinstance(error, ProviderRequestError):
        status_code = error.status_code
    if status_code == 401:
        return "auth_failed"
    if status_code in {402, 432, 433}:
        return "payment_required"
    if status_code == 403:
        return "forbidden"
    if status_code == 429:
        return "rate_limited"
    if isinstance(status_code, int) and 500 <= status_code < 600:
        return "upstream_5xx"
    if is_provider_timeout_error(error):
        return "timeout"
    return "network_or_provider_error"


def _search_locale_metadata(
    options: SearchOptions | None,
    settings: Any,
    *,
    tier: str,
    producer_locale: JsonDict | None = None,
) -> JsonDict:
    requested = options.requested_locale() if options is not None else {}
    effective = (options or SearchOptions()).effective_locale(settings)
    persisted_applied = producer_locale.get("applied_fields") if producer_locale is not None else None
    if isinstance(persisted_applied, list):
        applied_fields = [
            field for field in ("country", "search_lang", "ui_language") if field in persisted_applied
        ]
    elif tier == "brave":
        applied_fields = [field for field in ("country", "search_lang", "ui_language") if field in effective]
    elif tier == "tavily":
        applied_fields = (
            ["country"]
            if effective.get("country") and tavily_country_name(effective.get("country"))
            else []
        )
    elif tier in {"exa", "firecrawl", "firecrawl_search"}:
        applied_fields = ["country"] if effective.get("country") else []
    else:
        applied_fields = []
    ignored_fields = [field for field in ("country", "search_lang", "ui_language") if field in requested and field not in applied_fields]
    reason_codes: list[str] = []
    for field_name in ignored_fields:
        if tier == "indexed_external":
            reason_codes.append("indexed_external_locale_not_applied")
            break
        if tier not in {"none", "cache"}:
            reason_codes.append(f"provider_unsupported_{field_name}")
    return {
        "requested": requested,
        "effective": effective,
        "applied_fields": applied_fields,
        "ignored_fields": ignored_fields,
        "reason_codes": reason_codes,
    }


def _search_option_metadata(
    options: SearchOptions | None,
    *,
    tier: str,
    producer_metadata: JsonDict | None = None,
) -> JsonDict:
    """Describe which domain/freshness controls the producing tier applied."""

    requested = {
        "include_domains": list(options.include_domains) if options else [],
        "exclude_domains": list(options.exclude_domains) if options else [],
        "freshness": options.freshness if options else None,
    }
    requested_fields = [
        field_name
        for field_name, value in requested.items()
        if value not in (None, [], "")
    ]
    persisted_applied = (
        producer_metadata.get("applied_fields") if producer_metadata is not None else None
    )
    if isinstance(persisted_applied, list):
        applied_fields = [
            field_name for field_name in requested_fields if field_name in persisted_applied
        ]
    elif tier in {"brave", "tavily", "exa"}:
        applied_fields = list(requested_fields)
    elif tier in {"firecrawl", "firecrawl_search"}:
        applied_fields = [
            field_name
            for field_name in requested_fields
            if field_name in {"include_domains", "exclude_domains"}
        ]
    else:
        applied_fields = []
    ignored_fields = [
        field_name for field_name in requested_fields if field_name not in applied_fields
    ]
    reason_codes: list[str] = []
    for field_name in ignored_fields:
        if tier == "indexed_external":
            reason_codes.append("indexed_external_search_options_not_applied")
            break
        if tier == "none":
            reason_codes.append("no_provider_applied_search_options")
            break
        if tier not in {"cache"}:
            reason_codes.append(f"provider_unsupported_{field_name}")
    return {
        "requested": requested,
        "applied_fields": applied_fields,
        "ignored_fields": ignored_fields,
        "reason_codes": list(dict.fromkeys(reason_codes)),
    }


_TEMPORAL_ALIASES = {
    "published_at": ("published_at", "publishedAt", "published_date", "publishedDate", "datePublished"),
    "updated_at": ("updated_at", "updatedAt", "updated_date", "updatedDate", "lastModified"),
    "effective_at": ("effective_at", "effectiveAt", "effective_date", "effectiveDate", "validFrom"),
    "retrieved_at": ("retrieved_at", "retrievedAt"),
}


def _temporal_metadata(item: JsonDict) -> JsonDict:
    """Normalize exact provider timestamps; relative age stays descriptive only."""

    nested = item.get("metadata")
    sources = (item, nested if isinstance(nested, dict) else {})
    normalized: JsonDict = {}
    for canonical, aliases in _TEMPORAL_ALIASES.items():
        raw = next(
            (
                source.get(alias)
                for source in sources
                for alias in aliases
                if source.get(alias) not in (None, "")
            ),
            None,
        )
        timestamp = _exact_timestamp(raw)
        if timestamp is not None:
            normalized[canonical] = timestamp
    age = next((source.get("source_age") or source.get("age") for source in sources if source), None)
    if age:
        normalized["source_age"] = str(age)
    normalized.setdefault("retrieved_at", time.time())
    return normalized


def _exact_timestamp(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        timestamp = float(value)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            timestamp = float(text)
        except ValueError:
            try:
                parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            timestamp = parsed.timestamp()
    if not math.isfinite(timestamp) or timestamp < 0:
        return None
    if timestamp >= 100_000_000_000:
        timestamp /= 1000.0
    return timestamp


def _parse_embeddings(data: JsonDict, *, expected: int) -> list[list[float]]:
    """Parse a complete OpenAI-style batch without permitting positional drift."""

    items = data.get("data", []) if isinstance(data, dict) else []
    if not isinstance(items, list) or len(items) != expected:
        raise BaselineError(
            ErrorCode.UPSTREAM_ERROR,
            "embeddings response count did not match the request",
        )
    indexed = any(isinstance(item, dict) and "index" in item for item in items)
    if indexed:
        ordered: list[dict | None] = [None] * expected
        for item in items:
            if not isinstance(item, dict):
                raise BaselineError(
                    ErrorCode.UPSTREAM_ERROR,
                    "embeddings response mixed indexed and unindexed items",
                )
            index = item.get("index")
            if (
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= expected
                or ordered[index] is not None
            ):
                raise BaselineError(
                    ErrorCode.UPSTREAM_ERROR,
                    "embeddings response contained an invalid index mapping",
                )
            ordered[index] = item
        if any(item is None for item in ordered):
            raise BaselineError(
                ErrorCode.UPSTREAM_ERROR,
                "embeddings response index mapping was incomplete",
            )
        items = [item for item in ordered if item is not None]

    vectors: list[list[float]] = []
    try:
        for item in items:
            vector = item.get("embedding") if isinstance(item, dict) else None
            if not isinstance(vector, list):
                raise TypeError("embedding must be a list")
            vectors.append([float(value) for value in vector])
    except (TypeError, ValueError) as error:
        raise BaselineError(
            ErrorCode.UPSTREAM_ERROR,
            "embeddings response contained an invalid vector",
        ) from error
    return vectors


def default_retry_policy():
    """Expose the shared provider retry policy for adapter construction."""

    return retry_policy_from_env()
