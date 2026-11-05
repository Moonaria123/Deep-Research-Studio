"""Resolution of effective per-call LLM sampling/thinking parameters.

Centralizes the priority chain used across the orchestrator and adapters:

    effective_temperature = run.temperature      or model.temperature      or 0.2
    effective_effort      = run.thinking_effort  or model(thinking_budget)  or off

``thinking_effort`` is a coarse Off/Low/Medium/High dial that maps to a token
budget (SRB-aligned thresholds). The budget is what gets persisted on a model
profile; the effort enum is the user-facing surface. At emit time the budget is
translated back to an OpenAI ``reasoning_effort`` string for OpenAI-compatible
providers, and silently dropped for providers (e.g. Ollama) that do not support it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from urllib.parse import urlparse

from ..providers_registry import ModelProfile

# User-facing thinking dial. ``off`` means "no extended reasoning".
ThinkingEffort = Literal["off", "low", "medium", "high"]
THINKING_EFFORTS: tuple[str, ...] = ("off", "low", "medium", "high")

# System fallback when neither the run nor the model specifies a temperature.
DEFAULT_TEMPERATURE = 0.2

# SRB-aligned effort <-> token-budget thresholds.
_LOW_BUDGET = 1024
_MEDIUM_BUDGET = 4096
_HIGH_BUDGET = 8192


def normalize_effort(value: Any) -> ThinkingEffort | None:
    """Return a valid ``ThinkingEffort`` or ``None`` when absent.

    Raises :class:`ValueError` for a non-empty value that is not a known effort
    so callers can surface a 400 to the client.
    """

    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "":
        return None
    if text not in THINKING_EFFORTS:
        raise ValueError(f"unknown thinking_effort '{value}'")
    return text  # type: ignore[return-value]


def effort_to_budget(effort: ThinkingEffort | None) -> int | None:
    """Map an effort enum to a thinking token budget (``None`` for off/absent)."""

    if effort == "low":
        return _LOW_BUDGET
    if effort == "medium":
        return _MEDIUM_BUDGET
    if effort == "high":
        return _HIGH_BUDGET
    return None


def budget_to_effort(budget: int | None) -> ThinkingEffort:
    """Map a thinking token budget back to an effort enum (for emission/UI default)."""

    if not budget or budget <= 0:
        return "off"
    if budget <= _LOW_BUDGET:
        return "low"
    if budget <= _MEDIUM_BUDGET:
        return "medium"
    return "high"


def clamp_temperature(value: Any) -> float | None:
    """Return ``value`` parsed and clamped to ``[0, 2]``, or ``None`` when absent."""

    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(2.0, parsed))


@dataclass(frozen=True)
class LLMCallParams:
    """Effective sampling/thinking parameters for a single run's LLM calls."""

    temperature: float = DEFAULT_TEMPERATURE
    thinking_budget: int | None = None

    @property
    def effort(self) -> ThinkingEffort:
        """Return the effort label corresponding to the resolved budget."""

        return budget_to_effort(self.thinking_budget)


def resolve_llm_call_params(
    model: ModelProfile | None,
    *,
    run_temperature: Any = None,
    run_thinking_effort: Any = None,
) -> LLMCallParams:
    """Resolve effective params from run overrides over model defaults over fallbacks.

    ``run_thinking_effort`` of ``"off"`` is an explicit user choice to disable
    extended reasoning and takes precedence over the model default.
    """

    run_temp = clamp_temperature(run_temperature)
    model_temp = clamp_temperature(model.temperature) if model is not None else None
    temperature = run_temp if run_temp is not None else (
        model_temp if model_temp is not None else DEFAULT_TEMPERATURE
    )

    run_effort = normalize_effort(run_thinking_effort)
    if run_effort is not None:
        thinking_budget = effort_to_budget(run_effort)
    else:
        thinking_budget = model.thinking_budget if model is not None else None

    return LLMCallParams(temperature=temperature, thinking_budget=thinking_budget)


ReasoningTransport = Literal["top_level", "extra_body", "none"]
StructuredOutputStrategy = Literal["native", "prompt_validated"]


@dataclass(frozen=True)
class LLMProviderCapabilities:
    """Secret-free capabilities used to resolve one concrete LLM call."""

    provider_identity: str
    provider_family: str
    endpoint_type: str = "chat_completions"
    reasoning_transport: ReasoningTransport = "none"
    structured_output_strategy: StructuredOutputStrategy = "prompt_validated"
    output_token_parameter: str = "max_tokens"
    shared_output_cap: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_identity": self.provider_identity,
            "provider_family": self.provider_family,
            "endpoint_type": self.endpoint_type,
            "reasoning_transport": self.reasoning_transport,
            "structured_output_strategy": self.structured_output_strategy,
            "output_token_parameter": self.output_token_parameter,
            "shared_output_cap": self.shared_output_cap,
        }


@dataclass(frozen=True)
class ResolvedLLMCallOptions:
    """Resolved provider payload fields plus auditable applied/ignored metadata."""

    capabilities: LLMProviderCapabilities
    requested: dict[str, Any]
    applied: dict[str, str]
    ignored: dict[str, str]
    reason_codes: tuple[str, ...]
    request_fields: dict[str, Any]

    @property
    def provider_identity(self) -> str:
        return self.capabilities.provider_identity

    @property
    def provider_family(self) -> str:
        return self.capabilities.provider_family

    @property
    def endpoint_type(self) -> str:
        return self.capabilities.endpoint_type

    @property
    def structured_output_strategy(self) -> StructuredOutputStrategy:
        return self.capabilities.structured_output_strategy

    def to_dict(self) -> dict[str, Any]:
        """Return diagnostics without serializing prompts or response schemas."""

        provider_output_cap = self.request_fields.get(self.capabilities.output_token_parameter)
        visible_output_target = self.requested.get("visible_output_tokens")
        reasoning_reserve_tokens = None
        if (
            self.capabilities.shared_output_cap
            and isinstance(provider_output_cap, int)
            and isinstance(visible_output_target, int)
        ):
            reasoning_reserve_tokens = max(0, provider_output_cap - visible_output_target)
        return {
            **self.capabilities.to_dict(),
            "requested": dict(self.requested),
            "applied": dict(self.applied),
            "ignored": dict(self.ignored),
            "reason_codes": list(self.reason_codes),
            "thinking_target": self.requested.get("thinking_effort"),
            "visible_output_target": visible_output_target,
            "reasoning_reserve_tokens": reasoning_reserve_tokens,
            "provider_output_cap": provider_output_cap,
        }


def resolve_llm_call_options(
    *,
    provider_type: str,
    provider_id: str = "",
    base_url: str = "",
    model: str = "",
    thinking_budget: int | None,
    max_visible_tokens: int,
    response_format: dict[str, Any] | None,
) -> ResolvedLLMCallOptions:
    """Resolve only parameters the selected endpoint truthfully supports.

    Poe Chat Completions accepts custom bot parameters inside ``extra_body`` while
    silently ignoring ordinary top-level ``reasoning_effort`` and
    ``response_format``.  Keeping that distinction here prevents diagnostics from
    claiming a parameter was applied when the endpoint ignored it.
    """

    family = _provider_family(provider_type=provider_type, provider_id=provider_id, base_url=base_url)
    identity = str(provider_id or provider_type or family).strip().lower() or family
    if family == "poe" and identity in {"", "openai", "openai-compatible"}:
        identity = "poe"
    effort = budget_to_effort(thinking_budget)
    requested = {
        "thinking_effort": effort,
        "visible_output_tokens": max(1, int(max_visible_tokens)),
        "structured_output": bool(response_format),
    }
    output_token_parameter = (
        "max_completion_tokens"
        if family == "openai" and _openai_uses_completion_token_cap(model)
        else "max_tokens"
    )
    shared_output_cap = (
        True
        if family == "poe" or (family == "openai" and _openai_reasoning_supported(model))
        else False
        if family in {"openai", "ollama"}
        else None
    )
    visible_output_tokens = max(1, int(max_visible_tokens))
    reasoning_reserve_tokens = (
        int(effort_to_budget(effort) or 0)
        if effort != "off" and shared_output_cap is True
        else 0
    )
    provider_output_cap = visible_output_tokens + reasoning_reserve_tokens
    applied: dict[str, str] = {"visible_output_limit": output_token_parameter}
    ignored: dict[str, str] = {}
    reason_codes: list[str] = []
    request_fields: dict[str, Any] = {
        output_token_parameter: provider_output_cap
    }
    if reasoning_reserve_tokens:
        applied["reasoning_reserve"] = output_token_parameter
        reason_codes.append("shared_output_cap_includes_reasoning_reserve")

    if effort != "off":
        if family == "poe":
            request_fields["extra_body"] = {"reasoning_effort": effort}
            applied["reasoning_effort"] = "extra_body"
            reason_codes.append("poe_custom_parameter_best_effort")
        elif family == "openai" and _openai_reasoning_supported(model):
            request_fields["reasoning_effort"] = effort
            applied["reasoning_effort"] = "top_level"
        elif family == "litellm":
            ignored["reasoning_effort"] = "litellm_capability_unconfirmed"
            reason_codes.append("reasoning_capability_unconfirmed")
        else:
            ignored["reasoning_effort"] = "unsupported_or_unconfirmed"
            reason_codes.append("reasoning_capability_unconfirmed")

    if response_format:
        if family == "openai":
            request_fields["response_format"] = response_format
            applied["response_format"] = "native"
        elif family == "litellm":
            # LiteLLM's selected downstream model is not known at this layer.  The
            # trusted prompt + deterministic validator remains the safe contract.
            ignored["response_format"] = "litellm_capability_unconfirmed"
            reason_codes.append("structured_output_prompt_validated")
        elif family == "poe":
            ignored["response_format"] = "unsupported_by_poe_chat"
            reason_codes.append("structured_output_prompt_validated")
        else:
            ignored["response_format"] = "unsupported_or_unconfirmed"
            reason_codes.append("structured_output_prompt_validated")

    reasoning_transport: ReasoningTransport = (
        "extra_body"
        if family == "poe"
        else "top_level"
        if family == "openai" and _openai_reasoning_supported(model)
        else "none"
    )
    capabilities = LLMProviderCapabilities(
        provider_identity=identity,
        provider_family=family,
        reasoning_transport=reasoning_transport,
        structured_output_strategy="native" if family == "openai" else "prompt_validated",
        output_token_parameter=output_token_parameter,
        shared_output_cap=shared_output_cap,
    )
    return ResolvedLLMCallOptions(
        capabilities=capabilities,
        requested=requested,
        applied=applied,
        ignored=ignored,
        reason_codes=tuple(dict.fromkeys(reason_codes)),
        request_fields=request_fields,
    )


def _provider_family(*, provider_type: str, provider_id: str, base_url: str) -> str:
    provider = str(provider_id or "").strip().lower()
    configured_type = str(provider_type or "").strip().lower()
    try:
        host = str(urlparse(str(base_url or "")).hostname or "").lower()
    except ValueError:
        host = ""
    if provider == "poe" or host == "api.poe.com":
        return "poe"
    if configured_type == "ollama" or provider == "ollama":
        return "ollama"
    if configured_type == "litellm" or provider == "litellm":
        return "litellm"
    if configured_type and configured_type not in {"openai", "openai-compatible"}:
        return configured_type
    return "openai"


def _openai_reasoning_supported(model: str) -> bool:
    """Conservatively identify models with an explicit OpenAI reasoning contract."""

    normalized = str(model or "").strip().lower()
    if not normalized:
        # Legacy profiles did not always pass the model into parameter resolution.
        # Preserve their OpenAI behavior while new runtime calls pass it explicitly.
        return True
    return normalized.startswith(("o1", "o3", "o4", "gpt-5"))


def _openai_uses_completion_token_cap(model: str) -> bool:
    normalized = str(model or "").strip().lower()
    return normalized.startswith(("o1", "o3", "o4", "gpt-5"))


def thinking_payload_kwargs(provider_type: str, thinking_budget: int | None) -> dict[str, Any]:
    """Return chat-payload kwargs that carry the thinking signal for a provider.

    OpenAI-compatible providers receive ``reasoning_effort``; providers that do not
    support extended reasoning (Ollama) receive nothing. ``thinking_budget`` of
    ``None``/``0`` always yields an empty dict.
    """

    if not thinking_budget or thinking_budget <= 0:
        return {}
    if str(provider_type or "").strip().lower() == "ollama":
        return {}
    return {"reasoning_effort": budget_to_effort(thinking_budget)}
