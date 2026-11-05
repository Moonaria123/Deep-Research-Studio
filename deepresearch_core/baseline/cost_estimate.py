"""Pre-run cost estimates for research runs.

The estimator is deliberately conservative: it never calls an upstream LLM or
search provider, and it returns ``None`` for monetary fields when pricing is not
known locally. SQLite data remains untouched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

from .budget import (
    COUNCIL_SIZE_DEFAULT,
    COUNCIL_SIZE_MAX,
    COUNCIL_SIZE_MIN,
    COUNCIL_STRATEGIES,
    COUNCIL_STRATEGY_DEFAULT,
    Budget,
    Mode,
    preset_for,
)
from .errors import BaselineError, ErrorCode
from .search_options import SearchOptions, normalize_search_options

JsonDict = dict[str, Any]

_BUDGET_FIELDS = {"max_searches", "max_think_tokens", "max_output_tokens", "concurrency", "timeout_s", "top_k"}
_BRAVE_SEARCH_USD_PER_REQUEST = 5.0 / 1000.0
_FIRECRAWL_SEARCH_CREDITS_PER_10_RESULTS = 2.0

# Static fallback table is intentionally tiny. Prefer LiteLLM pricing when
# installed; use this only for well-known public OpenAI models documented in the
# official pricing/model pages.
_STATIC_LLM_USD_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
}

_INPUT_TOKENS_BY_MODE = {
    Mode.FAST: 2_000,
    Mode.MULTIPLE: 4_000,
    Mode.DEEP: 8_000,
    Mode.COUNCIL: 4_000,
}


@dataclass(frozen=True)
class RunCostEstimateRequest:
    """Normalized draft request for a pre-submit cost estimate.

    Unlike :class:`RunRequest`, this shape deliberately does not require
    ``query`` because Composer should refresh the preview before the user has
    written or submitted a prompt.
    """

    mode: Mode = Mode.FAST
    budget_overrides: dict[str, int] = field(default_factory=dict)
    llm_model: str = ""
    council_models: tuple[str, ...] = ()
    council_size: int = COUNCIL_SIZE_DEFAULT
    council_strategy: str = COUNCIL_STRATEGY_DEFAULT
    search_options: SearchOptions | None = None

    @staticmethod
    def from_payload(payload: JsonDict | None) -> "RunCostEstimateRequest":
        data = payload if isinstance(payload, dict) else {}
        mode_value = str(data.get("mode") or Mode.FAST.value).lower()
        try:
            mode = Mode(mode_value)
        except ValueError as error:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                f"unknown mode '{mode_value}'",
                details={"allowed": [m.value for m in Mode]},
            ) from error
        council_size = _clamp_council_size(data.get("council_size"))
        council_strategy = str(data.get("council_strategy") or COUNCIL_STRATEGY_DEFAULT).lower()
        if council_strategy not in COUNCIL_STRATEGIES:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                f"unknown council_strategy '{council_strategy}'",
                details={"allowed": list(COUNCIL_STRATEGIES)},
            )
        try:
            search_options = normalize_search_options(data.get("search_options"))
        except ValueError as error:
            raise BaselineError(ErrorCode.INVALID_REQUEST, str(error)) from error
        return RunCostEstimateRequest(
            mode=mode,
            budget_overrides=_normalize_budget_overrides(data.get("budget")),
            llm_model=str(data.get("llm_model") or "").strip(),
            council_models=tuple(str(m).strip() for m in (data.get("council_models") or []) if str(m).strip()),
            council_size=council_size,
            council_strategy=council_strategy,
            search_options=search_options,
        )

    def resolved_budget(self) -> Budget:
        base = preset_for(self.mode)
        return replace(base, **self.budget_overrides) if self.budget_overrides else base


@dataclass(frozen=True)
class CostEstimate:
    """Secret-free estimate returned by ``POST /runs/estimate-cost``."""

    mode: str
    searches: int
    recovery_searches_max: int
    search_usd_low: float | None
    search_usd_high: float | None
    search_credits_low: float | None
    search_credits_high: float | None
    llm_input_tokens_est: int
    llm_output_tokens_est: int
    llm_usd_low: float | None
    llm_usd_high: float | None
    total_usd_low: float | None
    total_usd_high: float | None
    confidence: str
    pricing_basis: JsonDict
    disclaimers: tuple[str, ...] = ()

    def to_dict(self) -> JsonDict:
        return {
            "mode": self.mode,
            "searches": self.searches,
            "recovery_searches_max": self.recovery_searches_max,
            "search_usd_low": self.search_usd_low,
            "search_usd_high": self.search_usd_high,
            "search_credits_low": self.search_credits_low,
            "search_credits_high": self.search_credits_high,
            "llm_input_tokens_est": self.llm_input_tokens_est,
            "llm_output_tokens_est": self.llm_output_tokens_est,
            "llm_usd_low": self.llm_usd_low,
            "llm_usd_high": self.llm_usd_high,
            "total_usd_low": self.total_usd_low,
            "total_usd_high": self.total_usd_high,
            "confidence": self.confidence,
            "pricing_basis": dict(self.pricing_basis),
            "disclaimers": list(self.disclaimers),
        }


def estimate_run_cost(
    request: RunCostEstimateRequest,
    *,
    model_id: str = "",
    provider_type: str = "",
    search_provider: str = "none",
) -> CostEstimate:
    """Return a bounded, secret-free pre-run cost estimate."""

    budget = request.resolved_budget()
    searches = max(0, int(budget.max_searches))
    recovery_searches_max = 0 if request.mode is Mode.COUNCIL else 1
    input_tokens, output_tokens = _estimate_llm_tokens(request, budget)
    if recovery_searches_max:
        input_tokens += _INPUT_TOKENS_BY_MODE[request.mode]
        output_tokens += int(budget.max_output_tokens)
    llm_low, llm_high, llm_basis, llm_disclaimer = estimate_llm_cost(
        model_id=model_id,
        provider_type=provider_type,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    search_low, search_high, credit_low, credit_high, search_basis, search_disclaimer = estimate_search_cost(
        search_provider=search_provider,
        searches=searches + recovery_searches_max,
        results_per_search=max(1, int(budget.max_searches)),
    )
    total_low = _sum_optional(search_low, llm_low)
    total_high = _sum_optional(search_high, llm_high)
    disclaimers = tuple(
        item
        for item in (
            "Estimates are approximate and may differ from provider billing.",
            llm_disclaimer,
            search_disclaimer,
        )
        if item
    )
    confidence = _confidence(llm_low, search_low, search_provider=search_provider, model_id=model_id)
    return CostEstimate(
        mode=request.mode.value,
        searches=searches,
        recovery_searches_max=recovery_searches_max,
        search_usd_low=search_low,
        search_usd_high=search_high,
        search_credits_low=credit_low,
        search_credits_high=credit_high,
        llm_input_tokens_est=input_tokens,
        llm_output_tokens_est=output_tokens,
        llm_usd_low=llm_low,
        llm_usd_high=llm_high,
        total_usd_low=total_low,
        total_usd_high=total_high,
        confidence=confidence,
        pricing_basis={
            "llm": llm_basis,
            "search": search_basis,
            "recovery": {
                "searches_max": recovery_searches_max,
                "rewrite_max": 1 if recovery_searches_max else 0,
                "conditional": True,
            },
        },
        disclaimers=disclaimers,
    )


def estimate_llm_cost(
    *,
    model_id: str,
    provider_type: str,
    input_tokens: int,
    output_tokens: int,
) -> tuple[float | None, float | None, JsonDict, str]:
    """Estimate LLM USD via LiteLLM or a small static public-price fallback."""

    model = str(model_id or "").strip()
    provider = str(provider_type or "").strip().lower()
    if provider == "ollama":
        return None, None, {"provider_type": provider, "model": model, "source": "local"}, "Local Ollama pricing is not estimated."
    if not model:
        return None, None, {"provider_type": provider or "unknown", "source": "unconfigured"}, "LLM model is not configured."
    litellm_cost = _estimate_with_litellm(model, input_tokens=input_tokens, output_tokens=output_tokens)
    if litellm_cost is not None:
        return litellm_cost, litellm_cost, {"provider_type": provider or "unknown", "model": model, "source": "litellm"}, ""
    static = _STATIC_LLM_USD_PER_1M.get(model.lower())
    if static:
        in_rate, out_rate = static
        total = (input_tokens / 1_000_000.0 * in_rate) + (output_tokens / 1_000_000.0 * out_rate)
        return round(total, 8), round(total, 8), {"provider_type": provider or "unknown", "model": model, "source": "static_openai"}, ""
    return None, None, {"provider_type": provider or "unknown", "model": model, "source": "unknown"}, "LLM pricing is unavailable for this model."


def estimate_search_cost(
    *,
    search_provider: str,
    searches: int,
    results_per_search: int,
) -> tuple[float | None, float | None, float | None, float | None, JsonDict, str]:
    provider = str(search_provider or "none").lower()
    count = max(0, int(searches))
    if provider == "brave":
        cost = round(count * _BRAVE_SEARCH_USD_PER_REQUEST, 8)
        return cost, cost, None, None, {"provider": "brave", "source": "brave_search_api"}, ""
    if provider == "firecrawl":
        blocks = max(1, math.ceil(max(1, int(results_per_search)) / 10))
        credits = round(count * blocks * _FIRECRAWL_SEARCH_CREDITS_PER_10_RESULTS, 4)
        return None, None, credits, credits, {"provider": "firecrawl", "source": "firecrawl_credits"}, "Firecrawl is shown in credits because USD conversion depends on the plan."
    if provider == "brave_firecrawl":
        # Runtime prefers Brave and falls back to Firecrawl; show the USD primary
        # path plus the possible fallback credit exposure.
        cost = round(count * _BRAVE_SEARCH_USD_PER_REQUEST, 8)
        blocks = max(1, math.ceil(max(1, int(results_per_search)) / 10))
        credits = round(count * blocks * _FIRECRAWL_SEARCH_CREDITS_PER_10_RESULTS, 4)
        return cost, cost, credits, credits, {"provider": "brave_firecrawl", "source": "brave_primary_firecrawl_fallback"}, "Firecrawl fallback is shown in credits because USD conversion depends on the plan."
    if provider == "mixed_provider_unknown":
        return (
            None,
            None,
            None,
            None,
            {"provider": "mixed_provider_unknown", "source": "cost_unavailable"},
            "Search cost unavailable for the configured mixed-provider chain.",
        )
    return None, None, None, None, {"provider": "none", "source": "unconfigured"}, "Search provider pricing is unavailable."


def _estimate_llm_tokens(request: RunCostEstimateRequest, budget: Budget) -> tuple[int, int]:
    input_tokens = _INPUT_TOKENS_BY_MODE[request.mode]
    output_tokens = int(budget.max_output_tokens)
    if request.mode is Mode.COUNCIL:
        size = max(COUNCIL_SIZE_MIN, min(COUNCIL_SIZE_MAX, int(request.council_size or COUNCIL_SIZE_DEFAULT)))
        if request.council_strategy == "parallel_vote":
            rounds = size
        elif request.council_strategy == "debate_once":
            rounds = size * 2 + 1
        else:
            rounds = size + 1
        input_tokens *= rounds
        output_tokens *= rounds
    return max(0, input_tokens), max(0, output_tokens)


def _estimate_with_litellm(model_id: str, *, input_tokens: int, output_tokens: int) -> float | None:
    try:
        import litellm  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001
        return None
    try:
        cost_per_token = getattr(litellm, "cost_per_token", None)
        if callable(cost_per_token):
            prompt_cost, completion_cost = cost_per_token(
                model=model_id,
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
            )
            return round(float(prompt_cost) + float(completion_cost), 8)
        completion_cost = getattr(litellm, "completion_cost", None)
        if callable(completion_cost):
            return round(
                float(
                    completion_cost(
                        model=model_id,
                        prompt_tokens=input_tokens,
                        completion_tokens=output_tokens,
                    )
                ),
                8,
            )
    except Exception:  # noqa: BLE001 - unknown model/pricing should not fail the preview
        return None
    return None


def _normalize_budget_overrides(value: Any) -> dict[str, int]:
    if value in (None, ""):
        return {}
    if not isinstance(value, dict):
        raise BaselineError(ErrorCode.INVALID_REQUEST, "budget must be an object")
    out: dict[str, int] = {}
    for key, raw in value.items():
        if key not in _BUDGET_FIELDS or raw in (None, ""):
            continue
        try:
            number = int(raw)
        except (TypeError, ValueError) as error:
            raise BaselineError(ErrorCode.INVALID_REQUEST, f"budget.{key} must be an integer") from error
        if number <= 0:
            raise BaselineError(ErrorCode.INVALID_REQUEST, f"budget.{key} must be positive")
        out[key] = number
    return out


def _clamp_council_size(value: Any) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError):
        return COUNCIL_SIZE_DEFAULT
    return max(COUNCIL_SIZE_MIN, min(COUNCIL_SIZE_MAX, size))


def _sum_optional(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    return round(first + second, 8)


def _confidence(
    llm_usd: float | None,
    search_usd: float | None,
    *,
    search_provider: str,
    model_id: str,
) -> str:
    if not model_id and search_provider == "none":
        return "unknown"
    if llm_usd is not None and search_usd is not None:
        return "medium"
    if llm_usd is not None or search_usd is not None:
        return "low"
    return "unknown"
