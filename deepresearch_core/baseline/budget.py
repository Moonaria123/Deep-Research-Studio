"""Run request + four-mode budget presets (baseline §四模式).

A :class:`Budget` bounds searches, thinking/output tokens, concurrency, and timeout.
Each mode has a default preset; callers may override individual fields per run.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from .search_options import SearchOptions, normalize_search_options


class Mode(str, Enum):
    """The four manual research modes."""

    FAST = "fast"
    MULTIPLE = "multiple"
    DEEP = "deep"
    COUNCIL = "council"


@dataclass(frozen=True)
class Budget:
    """Per-run resource budget."""

    max_searches: int
    max_think_tokens: int
    max_output_tokens: int
    concurrency: int
    timeout_s: int
    top_k: int = 8

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view."""

        return {
            "max_searches": self.max_searches,
            "max_think_tokens": self.max_think_tokens,
            "max_output_tokens": self.max_output_tokens,
            "concurrency": self.concurrency,
            "timeout_s": self.timeout_s,
            "top_k": self.top_k,
        }


@dataclass(frozen=True)
class RetrievalBudget:
    """Internal per-mode candidate, fetch, and downstream reserve budget."""

    results_per_query: int
    fetches_max: int
    fetch_concurrency: int
    writer_reserve_s: float
    finalize_reserve_s: float
    cleanup_reserve_s: float
    recovery_fetches_max: int = 0
    candidate_count: int = 0
    fetches_used: int = 0
    recovery_fetches_used: int = 0

    @property
    def reserve_total_s(self) -> float:
        """Return the minimum time reserved after retrieval starts."""

        return (
            max(0.0, float(self.writer_reserve_s))
            + max(0.0, float(self.finalize_reserve_s))
            + max(0.0, float(self.cleanup_reserve_s))
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable run-result observability shape."""

        return {
            "results_per_query": int(self.results_per_query),
            "candidate_count": int(self.candidate_count),
            "fetches_max": int(self.fetches_max),
            "fetches_used": int(self.fetches_used),
            "fetch_concurrency": int(self.fetch_concurrency),
            "writer_reserve_s": float(self.writer_reserve_s),
            "finalize_reserve_s": float(self.finalize_reserve_s),
            "cleanup_reserve_s": float(self.cleanup_reserve_s),
            "recovery_fetches_max": int(self.recovery_fetches_max),
            "recovery_fetches_used": int(self.recovery_fetches_used),
        }


_PRESETS: dict[Mode, Budget] = {
    Mode.FAST: Budget(max_searches=1, max_think_tokens=256, max_output_tokens=700, concurrency=1, timeout_s=30, top_k=6),
    Mode.MULTIPLE: Budget(max_searches=4, max_think_tokens=512, max_output_tokens=1200, concurrency=3, timeout_s=60, top_k=8),
    Mode.DEEP: Budget(max_searches=6, max_think_tokens=1024, max_output_tokens=3600, concurrency=3, timeout_s=180, top_k=10),
    Mode.COUNCIL: Budget(max_searches=4, max_think_tokens=768, max_output_tokens=1400, concurrency=3, timeout_s=120, top_k=8),
}

_RETRIEVAL_PRESETS: dict[Mode, RetrievalBudget] = {
    Mode.FAST: RetrievalBudget(6, 1, 1, 8.0, 2.0, 2.0, recovery_fetches_max=1),
    Mode.MULTIPLE: RetrievalBudget(6, 4, 3, 18.0, 3.0, 2.0, recovery_fetches_max=2),
    Mode.DEEP: RetrievalBudget(8, 8, 3, 40.0, 5.0, 2.0, recovery_fetches_max=2),
    Mode.COUNCIL: RetrievalBudget(8, 4, 3, 25.0, 3.0, 2.0),
}

_PREVIEW_FIELDS = ("max_searches", "max_output_tokens", "timeout_s", "top_k")


def preset_for(mode: Mode) -> Budget:
    """Return the default budget preset for a mode."""

    return _PRESETS[mode]


def retrieval_budget_for(mode: Mode | str) -> RetrievalBudget:
    """Return the immutable T49 retrieval budget for a research mode."""

    from .errors import BaselineError, ErrorCode

    try:
        normalized = mode if isinstance(mode, Mode) else Mode(str(mode).lower())
    except ValueError as error:
        raise BaselineError(ErrorCode.INVALID_REQUEST, f"unknown mode '{mode}'") from error
    return _RETRIEVAL_PRESETS[normalized]


def validate_deadline_budget(budget: Budget, retrieval_budget: RetrievalBudget) -> None:
    """Reject a run timeout that cannot cover its locked downstream reserves."""

    from .errors import BaselineError, ErrorCode

    try:
        timeout_s = float(budget.timeout_s)
    except (TypeError, ValueError) as error:
        raise BaselineError(ErrorCode.INVALID_REQUEST, "timeout_s must be a number") from error
    if timeout_s <= 0:
        raise BaselineError(ErrorCode.INVALID_REQUEST, "timeout_s must be positive")
    required_s = retrieval_budget.reserve_total_s
    if timeout_s < required_s:
        raise BaselineError(
            ErrorCode.INVALID_REQUEST,
            "timeout_s is smaller than the mode's required deadline reserves",
            details={"timeout_s": timeout_s, "required_s": required_s},
        )


def mode_budget_presets() -> dict[str, dict[str, int]]:
    """Return the mode budget fields consumed by the frontend preview UI."""

    return {
        mode.value: {field: int(preset.to_dict()[field]) for field in _PREVIEW_FIELDS}
        for mode, preset in _PRESETS.items()
    }


# Council member count bounds (baseline §委员会模式: 2..5, default 3).
COUNCIL_SIZE_MIN = 2
COUNCIL_SIZE_MAX = 5
COUNCIL_SIZE_DEFAULT = 3

# Council aggregation strategies.
#   parallel_vote   -> drafts -> best-covered member answer, no judge.
#   judge_synthesis -> drafts -> judge (current default).
#   debate_once     -> drafts -> one critique round -> judge.
COUNCIL_STRATEGY_DEFAULT = "judge_synthesis"
COUNCIL_STRATEGIES = ("parallel_vote", "judge_synthesis", "debate_once")

# Pre-run long-term memory injection policy (DR-ITER-054).
#   off            -> never recall long-term memory (backend default; never implicit).
#   workspace      -> recall workspace-layer memories only.
#   workspace_user -> recall workspace + user layer memories.
# The "default on for a session" decision is made by the frontend, which sends an
# explicit policy; the backend never infers a non-off policy.
MEMORY_POLICY_DEFAULT = "off"
MEMORY_POLICIES = ("off", "workspace", "workspace_user")
MEMORY_SCOPE_DEFAULT = "default"

# Runtime Library retrieval scope (DR-ITER-060 T9b).
# Library context is opt-in and non-citable; the default preserves baseline behavior.
LIBRARY_SCOPE_DEFAULT = "off"
LIBRARY_SCOPES = ("off", "notes", "reports", "all")


def _normalize_context_budget(value: Any) -> dict[str, Any] | None:
    """Validate + normalize a per-run ``context_budget`` override (DR-ITER-057).

    Accepts ``None`` (no override), a named preset, custom shares, or a resolved
    caps dict. Returns the normalized ``to_dict`` form so the worker round-trips a
    consistent shape. Invalid payloads raise ``INVALID_REQUEST``.
    """

    if value is None:
        return None
    from .errors import BaselineError, ErrorCode
    from .context_budget import coerce_payload

    try:
        return coerce_payload(value).to_dict()
    except ValueError as error:
        raise BaselineError(ErrorCode.INVALID_REQUEST, str(error)) from error


def _clamp_council_size(value: Any) -> int:
    """Clamp a requested council size into ``[COUNCIL_SIZE_MIN, COUNCIL_SIZE_MAX]``."""

    try:
        size = int(value)
    except (TypeError, ValueError):
        return COUNCIL_SIZE_DEFAULT
    return max(COUNCIL_SIZE_MIN, min(COUNCIL_SIZE_MAX, size))


def _locale_snapshot(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): str(item)
        for key, item in value.items()
        if key in {"country", "search_lang", "ui_language"} and item not in (None, "")
    }


@dataclass(frozen=True)
class RunRequest:
    """A normalized request to execute a research run."""

    query: str
    mode: Mode = Mode.FAST
    session_id: str = ""
    budget: Budget | None = None
    llm_model: str = ""
    council_models: list[str] = field(default_factory=list)
    council_size: int = COUNCIL_SIZE_DEFAULT
    council_strategy: str = COUNCIL_STRATEGY_DEFAULT
    temperature: float | None = None
    thinking_effort: str | None = None
    memory_policy: str = MEMORY_POLICY_DEFAULT
    workspace_key: str = MEMORY_SCOPE_DEFAULT
    user_key: str = MEMORY_SCOPE_DEFAULT
    library_scope: str = LIBRARY_SCOPE_DEFAULT
    context_budget: dict[str, Any] | None = None
    search_options: SearchOptions | None = None
    requested_locale: dict[str, str] = field(default_factory=dict)
    effective_locale: dict[str, str] = field(default_factory=dict)
    resolved_search_usage: dict[str, Any] = field(default_factory=dict)
    requested_at: float = 0.0
    deadline_at: float = 0.0
    auto_checkpoint_enabled: bool = False
    kg_aware_retrieval_enabled: bool = False
    kg_retrieval_snapshot: dict[str, Any] = field(default_factory=dict)
    checkpoint_parent_run_id: str = ""
    checkpoint_revision: int = 0
    lineage_id: str = ""
    checkpoint_action: str = ""
    checkpoint_remaining_plan: dict[str, Any] | None = None
    checkpoint_incremental_budget: dict[str, Any] | None = None
    overrides: dict[str, Any] = field(default_factory=dict)

    def resolved_budget(self) -> Budget:
        """Return the effective budget (preset merged with overrides)."""

        base = self.budget or preset_for(self.mode)
        if not self.overrides:
            return base
        allowed = {k: v for k, v in self.overrides.items() if k in base.to_dict()}
        return replace(base, **allowed) if allowed else base

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-serializable payload that round-trips through ``from_payload``.

        Used to hand a run off to an out-of-process job worker (RQ). Only the fields
        accepted by ``from_payload`` are emitted; ``budget`` carries the raw overrides.
        """

        return {
            "query": self.query,
            "mode": self.mode.value,
            "session_id": self.session_id,
            "llm_model": self.llm_model,
            "council_models": list(self.council_models),
            "council_size": self.council_size,
            "council_strategy": self.council_strategy,
            "temperature": self.temperature,
            "thinking_effort": self.thinking_effort,
            "memory_policy": self.memory_policy,
            "workspace_key": self.workspace_key,
            "user_key": self.user_key,
            "library_scope": self.library_scope,
            "context_budget": dict(self.context_budget) if self.context_budget else None,
            "search_options": self.search_options.to_dict() if self.search_options else None,
            "requested_locale": dict(self.requested_locale),
            "effective_locale": dict(self.effective_locale),
            "resolved_search_usage": dict(self.resolved_search_usage),
            "requested_at": float(self.requested_at),
            "deadline_at": float(self.deadline_at),
            "auto_checkpoint_enabled": bool(self.auto_checkpoint_enabled),
            "kg_aware_retrieval_enabled": bool(self.kg_aware_retrieval_enabled),
            "kg_retrieval_snapshot": dict(self.kg_retrieval_snapshot),
            "checkpoint_parent_run_id": self.checkpoint_parent_run_id,
            "checkpoint_revision": int(self.checkpoint_revision),
            "lineage_id": self.lineage_id,
            "checkpoint_action": self.checkpoint_action,
            "checkpoint_remaining_plan": (
                dict(self.checkpoint_remaining_plan)
                if self.checkpoint_remaining_plan is not None
                else None
            ),
            "checkpoint_incremental_budget": (
                dict(self.checkpoint_incremental_budget)
                if self.checkpoint_incremental_budget is not None
                else None
            ),
            "budget": dict(self.overrides),
        }

    @staticmethod
    def from_payload(payload: dict[str, Any]) -> "RunRequest":
        """Build a RunRequest from an API payload, validating the mode."""

        from .errors import BaselineError, ErrorCode
        from .llm_call_params import clamp_temperature, normalize_effort

        query = str(payload.get("query") or "").strip()
        if not query:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "query is required")
        mode_value = str(payload.get("mode") or "fast").lower()
        try:
            mode = Mode(mode_value)
        except ValueError as error:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                f"unknown mode '{mode_value}'",
                details={"allowed": [m.value for m in Mode]},
            ) from error
        council_size = (
            _clamp_council_size(payload.get("council_size"))
            if payload.get("council_size") is not None
            else COUNCIL_SIZE_DEFAULT
        )
        council_strategy = str(payload.get("council_strategy") or COUNCIL_STRATEGY_DEFAULT).lower()
        if council_strategy not in COUNCIL_STRATEGIES:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                f"unknown council_strategy '{council_strategy}'",
                details={"allowed": list(COUNCIL_STRATEGIES)},
            )
        temperature = clamp_temperature(payload.get("temperature"))
        try:
            thinking_effort = normalize_effort(payload.get("thinking_effort"))
        except ValueError as error:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                str(error),
                details={"allowed": ["off", "low", "medium", "high"]},
            ) from error
        memory_policy = str(payload.get("memory_policy") or MEMORY_POLICY_DEFAULT).lower()
        if memory_policy not in MEMORY_POLICIES:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                f"unknown memory_policy '{memory_policy}'",
                details={"allowed": list(MEMORY_POLICIES)},
            )
        library_scope = str(payload.get("library_scope") or LIBRARY_SCOPE_DEFAULT).lower()
        if library_scope not in LIBRARY_SCOPES:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                f"unknown library_scope '{library_scope}'",
                details={"allowed": list(LIBRARY_SCOPES)},
            )
        context_budget = _normalize_context_budget(payload.get("context_budget"))
        try:
            search_options = normalize_search_options(payload.get("search_options"))
        except ValueError as error:
            raise BaselineError(ErrorCode.INVALID_REQUEST, str(error)) from error
        requested_locale = _locale_snapshot(payload.get("requested_locale"))
        effective_locale = _locale_snapshot(payload.get("effective_locale"))
        resolved_search_usage = (
            dict(payload.get("resolved_search_usage") or {})
            if isinstance(payload.get("resolved_search_usage"), dict)
            else {}
        )
        try:
            requested_at = float(payload.get("requested_at") or 0.0)
        except (TypeError, ValueError):
            requested_at = 0.0
        try:
            deadline_at = float(payload.get("deadline_at") or 0.0)
        except (TypeError, ValueError):
            deadline_at = 0.0
        auto_checkpoint_enabled = payload.get("auto_checkpoint_enabled", False)
        if not isinstance(auto_checkpoint_enabled, bool):
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "auto_checkpoint_enabled must be a boolean",
            )
        kg_aware_retrieval_enabled = payload.get("kg_aware_retrieval_enabled", False)
        if not isinstance(kg_aware_retrieval_enabled, bool):
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "kg_aware_retrieval_enabled must be a boolean",
            )
        kg_retrieval_snapshot = payload.get("kg_retrieval_snapshot")
        if kg_retrieval_snapshot is None:
            kg_retrieval_snapshot = {}
        if not isinstance(kg_retrieval_snapshot, dict):
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "kg_retrieval_snapshot must be an object",
            )
        try:
            checkpoint_revision = int(payload.get("checkpoint_revision") or 0)
        except (TypeError, ValueError) as error:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "checkpoint_revision must be an integer",
            ) from error
        if checkpoint_revision < 0:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "checkpoint_revision must be non-negative",
            )
        checkpoint_action = str(payload.get("checkpoint_action") or "").strip().lower()
        if checkpoint_action not in {"", "continue", "revise"}:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "checkpoint_action must be continue or revise",
            )
        checkpoint_remaining_plan = payload.get("checkpoint_remaining_plan")
        if checkpoint_remaining_plan is not None and not isinstance(
            checkpoint_remaining_plan, dict
        ):
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "checkpoint_remaining_plan must be an object",
            )
        checkpoint_incremental_budget = payload.get("checkpoint_incremental_budget")
        if checkpoint_incremental_budget is not None and not isinstance(
            checkpoint_incremental_budget, dict
        ):
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "checkpoint_incremental_budget must be an object",
            )
        return RunRequest(
            query=query,
            mode=mode,
            session_id=str(payload.get("session_id") or ""),
            llm_model=str(payload.get("llm_model") or "").strip(),
            council_models=[str(m).strip() for m in (payload.get("council_models") or []) if str(m).strip()],
            council_size=council_size,
            council_strategy=council_strategy,
            temperature=temperature,
            thinking_effort=thinking_effort,
            memory_policy=memory_policy,
            workspace_key=str(payload.get("workspace_key") or MEMORY_SCOPE_DEFAULT),
            user_key=str(payload.get("user_key") or MEMORY_SCOPE_DEFAULT),
            library_scope=library_scope,
            context_budget=context_budget,
            search_options=search_options,
            requested_locale=requested_locale,
            effective_locale=effective_locale,
            resolved_search_usage=resolved_search_usage,
            requested_at=requested_at,
            deadline_at=deadline_at,
            auto_checkpoint_enabled=auto_checkpoint_enabled,
            kg_aware_retrieval_enabled=kg_aware_retrieval_enabled,
            kg_retrieval_snapshot=dict(kg_retrieval_snapshot),
            checkpoint_parent_run_id=str(
                payload.get("checkpoint_parent_run_id") or ""
            ).strip(),
            checkpoint_revision=checkpoint_revision,
            lineage_id=str(payload.get("lineage_id") or "").strip(),
            checkpoint_action=checkpoint_action,
            checkpoint_remaining_plan=(
                dict(checkpoint_remaining_plan)
                if isinstance(checkpoint_remaining_plan, dict)
                else None
            ),
            checkpoint_incremental_budget=(
                dict(checkpoint_incremental_budget)
                if isinstance(checkpoint_incremental_budget, dict)
                else None
            ),
            overrides=dict(payload.get("budget") or {}),
        )
