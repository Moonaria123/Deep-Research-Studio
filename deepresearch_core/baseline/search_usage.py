"""Deterministic search-provider usage policy (DR-ITER-146)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence


SearchProviderId = Literal["brave", "tavily", "exa", "firecrawl"]
SearchUsageMode = Literal["single", "dual_role", "auto"]

SEARCH_PROVIDERS: tuple[SearchProviderId, ...] = (
    "brave",
    "tavily",
    "exa",
    "firecrawl",
)
SEARCH_CAPABLE_PROVIDERS = frozenset(SEARCH_PROVIDERS)
EXTRACT_CAPABLE_PROVIDERS = frozenset({"tavily", "exa", "firecrawl"})
PROVIDER_LOCALE_FIELDS: dict[SearchProviderId, frozenset[str]] = {
    "brave": frozenset({"country", "search_lang", "ui_language"}),
    "tavily": frozenset({"country"}),
    "exa": frozenset({"country"}),
    "firecrawl": frozenset({"country"}),
}


@dataclass(frozen=True)
class SearchUsagePolicy:
    """User-configurable global or session search routing policy."""

    usage_mode: SearchUsageMode = "auto"
    primary_provider: SearchProviderId | None = None
    search_provider: SearchProviderId | None = None
    extract_provider: SearchProviderId | None = None
    fallback_chain: tuple[SearchProviderId, ...] = SEARCH_PROVIDERS

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "usage_mode": self.usage_mode,
            "fallback_chain": list(self.fallback_chain),
        }
        if self.primary_provider is not None:
            payload["primary_provider"] = self.primary_provider
        if self.search_provider is not None:
            payload["search_provider"] = self.search_provider
        if self.extract_provider is not None:
            payload["extract_provider"] = self.extract_provider
        return payload


DEFAULT_SEARCH_USAGE_POLICY = SearchUsagePolicy()


@dataclass(frozen=True)
class ResolvedSearchUsage:
    """Run-frozen search and extract provider chains."""

    source: Literal["global", "session"]
    effective_policy: SearchUsagePolicy
    search_chain: tuple[SearchProviderId, ...]
    extract_chain: tuple[SearchProviderId, ...]
    reason_codes: tuple[str, ...] = ()
    provider_config_revision: int | None = None
    credential_revision_snapshot: tuple[tuple[SearchProviderId, int], ...] = ()
    external_search_status: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "source": self.source,
            "effective_policy": self.effective_policy.to_dict(),
            "search_chain": list(self.search_chain),
            "extract_chain": list(self.extract_chain),
            "search_fallback_tail": ["cache", "indexed_external"],
            "reason_codes": list(self.reason_codes),
        }
        if self.provider_config_revision is not None:
            payload["provider_config_revision"] = self.provider_config_revision
        if self.credential_revision_snapshot:
            payload["credential_revision_snapshot"] = dict(self.credential_revision_snapshot)
        if self.external_search_status is not None:
            payload["external_search_status"] = dict(self.external_search_status)
        return payload


def normalize_search_usage_policy(
    value: object,
    *,
    require_complete: bool = False,
) -> SearchUsagePolicy:
    """Validate and normalize a public search usage policy."""

    if value is None:
        if require_complete:
            raise ValueError("search usage policy must be an object")
        return DEFAULT_SEARCH_USAGE_POLICY
    if isinstance(value, SearchUsagePolicy):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("search usage policy must be an object")

    allowed = {
        "usage_mode",
        "primary_provider",
        "search_provider",
        "extract_provider",
        "fallback_chain",
    }
    unknown = sorted(str(key) for key in set(value) - allowed)
    if unknown:
        raise ValueError(f"unsupported search usage fields: {', '.join(unknown)}")

    raw_mode = value.get("usage_mode")
    if raw_mode is None and require_complete:
        raise ValueError("usage_mode is required")
    mode = str(raw_mode or "auto").strip().lower()
    if mode not in {"single", "dual_role", "auto"}:
        raise ValueError("usage_mode must be single, dual_role, or auto")

    fallback_value = value.get("fallback_chain", SEARCH_PROVIDERS)
    if require_complete and "fallback_chain" not in value:
        raise ValueError("fallback_chain is required")
    if not isinstance(fallback_value, Sequence) or isinstance(fallback_value, (str, bytes)):
        raise ValueError("fallback_chain must be an array")
    fallback = tuple(_provider(item, field="fallback_chain") for item in fallback_value)
    if len(set(fallback)) != len(fallback):
        raise ValueError("fallback_chain contains duplicate providers")

    primary = _optional_provider(value.get("primary_provider"), field="primary_provider")
    search = _optional_provider(value.get("search_provider"), field="search_provider")
    extract = _optional_provider(value.get("extract_provider"), field="extract_provider")

    if mode == "single":
        if primary is None:
            raise ValueError("primary_provider is required for single mode")
        search = None
        extract = None
    elif mode == "dual_role":
        if search is None:
            raise ValueError("search_provider is required for dual_role mode")
        if extract is None:
            raise ValueError("extract_provider is required for dual_role mode")
        if extract not in EXTRACT_CAPABLE_PROVIDERS:
            raise ValueError(f"{extract} is not an extract-capable provider")
        primary = None
    else:
        primary = None
        search = None
        extract = None

    return SearchUsagePolicy(
        usage_mode=mode,  # type: ignore[arg-type]
        primary_provider=primary,
        search_provider=search,
        extract_provider=extract,
        fallback_chain=fallback,
    )


def resolve_search_usage(
    policy: SearchUsagePolicy | Mapping[str, Any],
    *,
    source: str,
    configured: Mapping[str, object],
    requested_locale: Mapping[str, str] | None = None,
    provider_config_revision: int | None = None,
) -> ResolvedSearchUsage:
    """Resolve a policy to deterministic, configured run-level chains."""

    normalized = normalize_search_usage_policy(policy)
    if source not in {"global", "session"}:
        raise ValueError("search usage source must be global or session")

    if normalized.usage_mode == "single":
        search_order = _dedupe((normalized.primary_provider, *normalized.fallback_chain))
        extract_order = search_order
    elif normalized.usage_mode == "dual_role":
        search_order = _dedupe((normalized.search_provider, *normalized.fallback_chain))
        extract_order = _dedupe((normalized.extract_provider, *normalized.fallback_chain))
    else:
        search_order = list(normalized.fallback_chain)
        requested = set(requested_locale or {}) & {"country", "search_lang", "ui_language"}
        if requested:
            positions = {provider: index for index, provider in enumerate(search_order)}
            search_order.sort(
                key=lambda provider: (
                    -len(PROVIDER_LOCALE_FIELDS[provider] & requested),
                    positions[provider],
                )
            )
        extract_order = list(normalized.fallback_chain)

    reasons: list[str] = []
    search_chain: list[SearchProviderId] = []
    extract_chain: list[SearchProviderId] = []
    for provider in search_order:
        reason = _provider_exclusion_reason(configured.get(provider), capability="search")
        if reason:
            reasons.append(f"{provider}_{reason}")
            continue
        search_chain.append(provider)
    for provider in extract_order:
        if provider not in EXTRACT_CAPABLE_PROVIDERS:
            reasons.append(f"{provider}_extract_unsupported")
            continue
        reason = _provider_exclusion_reason(configured.get(provider), capability="extract")
        if reason:
            reasons.append(f"{provider}_{reason}")
            continue
        extract_chain.append(provider)

    revisions: list[tuple[SearchProviderId, int]] = []
    for provider in SEARCH_PROVIDERS:
        value = configured.get(provider)
        if not isinstance(value, Mapping) or value.get("credential_source") != "config":
            continue
        try:
            revision = int(value.get("credential_revision") or 0)
        except (TypeError, ValueError):
            revision = 0
        revisions.append((provider, max(0, revision)))
    selected_providers = tuple(dict.fromkeys((*search_chain, *extract_chain)))
    selected_states = [configured.get(provider) for provider in selected_providers]
    all_available = bool(selected_states) and all(
        isinstance(value, Mapping) and value.get("probe_state") == "available"
        for value in selected_states
    )
    external_status = {
        "status": "ready" if search_chain and all_available else "warn" if search_chain else "unavailable",
        "runtime_ready": bool(search_chain),
        "reason_codes": list(dict.fromkeys(reasons)),
    }

    return ResolvedSearchUsage(
        source=source,  # type: ignore[arg-type]
        effective_policy=normalized,
        search_chain=tuple(search_chain),
        extract_chain=tuple(extract_chain),
        reason_codes=tuple(dict.fromkeys(reasons)),
        provider_config_revision=provider_config_revision,
        credential_revision_snapshot=tuple(revisions),
        external_search_status=external_status,
    )


def provider_capabilities(configured: Mapping[str, object]) -> dict[str, dict[str, Any]]:
    """Return the public secret-free capability map."""

    result: dict[str, dict[str, Any]] = {}
    for provider in SEARCH_PROVIDERS:
        value = configured.get(provider)
        is_configured = _configured(value, capability="configured")
        state = dict(value) if isinstance(value, Mapping) else {}
        result[provider] = {
            **state,
            "configured": is_configured,
            "search_ready": is_configured and provider in SEARCH_CAPABLE_PROVIDERS,
            "extract_ready": is_configured and provider in EXTRACT_CAPABLE_PROVIDERS,
        }
    return result


def _provider(value: object, *, field: str) -> SearchProviderId:
    provider = str(value or "").strip().lower()
    if provider not in SEARCH_PROVIDERS:
        raise ValueError(f"{field} contains unknown provider '{provider}'")
    return provider  # type: ignore[return-value]


def _optional_provider(value: object, *, field: str) -> SearchProviderId | None:
    if value is None or str(value).strip() == "":
        return None
    return _provider(value, field=field)


def _dedupe(values: Sequence[SearchProviderId | None]) -> list[SearchProviderId]:
    return list(dict.fromkeys(value for value in values if value is not None))


def _configured(value: object, *, capability: str) -> bool:
    if isinstance(value, Mapping):
        if capability == "search":
            return bool(value.get("search_ready", value.get("configured")))
        if capability == "extract":
            return bool(value.get("extract_ready", value.get("configured")))
        return bool(value.get("configured"))
    return bool(value)


def _provider_exclusion_reason(value: object, *, capability: str) -> str:
    if not isinstance(value, Mapping):
        return "not_configured" if not value else ""
    if not bool(value.get("configured")):
        return "not_configured"
    if capability == "extract" and value.get("extract_ready") is False:
        return "extract_unsupported"
    if value.get("enabled") is False:
        return "disabled"
    if value.get("runtime_eligible") is False:
        probe_state = str(value.get("probe_state") or "unavailable").strip()
        return probe_state or "unavailable"
    return ""
