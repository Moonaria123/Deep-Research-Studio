"""Secret-free External search provider state derivation (DR-ITER-147).

This module deliberately models only live External search providers. Retrieval
cache, ``indexed_external``, Library/local evidence, and Memory evidence are not
provider readiness inputs and must never be added to ``SEARCH_PROVIDER_IDS``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal, Mapping


SearchProviderId = Literal["brave", "tavily", "exa", "firecrawl"]
CredentialSource = Literal["none", "config", "secure", "env"]

SEARCH_PROVIDER_IDS: tuple[SearchProviderId, ...] = (
    "brave",
    "tavily",
    "exa",
    "firecrawl",
)
PROBE_STALE_AFTER_S = 24 * 60 * 60
SOFT_FAILURE_COOLDOWN_S = 60

_RUNNABLE_PROBE_STATES = frozenset({"available", "not_tested", "stale"})
_PERSISTABLE_PROBE_STATES = frozenset({"available", "rate_limited", "action_required"})
_PROBE_STATES = frozenset(
    {
        "not_configured",
        "not_tested",
        "stale",
        "available",
        "rate_limited",
        "action_required",
        "service_unavailable",
        "network_unreachable",
        "credential_rejected",
    }
)


@dataclass(frozen=True)
class SearchProviderState:
    """Derived public state for one live External search provider."""

    provider: SearchProviderId
    configured: bool
    credential_source: CredentialSource
    credential_revision: int | None
    enabled: bool
    probe_state: str
    credential_state: str
    availability: str
    lamp: str
    checked_at: float | None
    stale_at: float | None
    retry_after: float | None
    probe_credential_revision: int | None
    runtime_eligible: bool
    selectable: bool
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "configured": self.configured,
            "credential_source": self.credential_source,
            "credential_revision": self.credential_revision,
            "enabled": self.enabled,
            "probe_state": self.probe_state,
            "credential_state": self.credential_state,
            "availability": self.availability,
            "lamp": self.lamp,
            "checked_at": self.checked_at,
            "stale_at": self.stale_at,
            "retry_after": self.retry_after,
            "probe_credential_revision": self.probe_credential_revision,
            "runtime_eligible": self.runtime_eligible,
            "selectable": self.selectable,
            "reason_codes": list(self.reason_codes),
        }


def normalize_state_block(value: object) -> dict[str, Any]:
    """Return a safe in-memory state block without mutating legacy config."""

    raw = value if isinstance(value, Mapping) else {}
    try:
        revision = max(0, int(raw.get("revision") or 0))
    except (TypeError, ValueError):
        revision = 0
    raw_providers = raw.get("providers") if isinstance(raw.get("providers"), Mapping) else {}
    providers: dict[str, dict[str, Any]] = {}
    for provider in SEARCH_PROVIDER_IDS:
        item = raw_providers.get(provider) if isinstance(raw_providers, Mapping) else None
        item = item if isinstance(item, Mapping) else {}
        normalized: dict[str, Any] = {"enabled": bool(item.get("enabled", True))}
        credential_revision = _positive_int(item.get("credential_revision"))
        if credential_revision is not None:
            normalized["credential_revision"] = credential_revision
        last_probe = sanitize_probe_snapshot(item.get("last_probe"))
        if last_probe:
            normalized["last_probe"] = last_probe
        providers[provider] = normalized
    return {"revision": revision, "providers": providers}


def sanitize_probe_snapshot(value: object) -> dict[str, Any]:
    """Keep only secret-free, typed probe fields suitable for config storage."""

    if not isinstance(value, Mapping):
        return {}
    state = str(value.get("probe_state") or "").strip()
    if state not in _PROBE_STATES - {"not_configured", "not_tested", "stale"}:
        return {}
    snapshot: dict[str, Any] = {
        "probe_state": state,
        "credential_state": str(value.get("credential_state") or "unknown"),
        "availability": str(value.get("availability") or "unknown"),
        "checked_at": _nonnegative_float(value.get("checked_at")) or 0.0,
        "reason_codes": [
            str(item)[:80]
            for item in (value.get("reason_codes") or [])
            if isinstance(item, str) and item.strip()
        ][:12],
    }
    retry_after = _nonnegative_float(value.get("retry_after"))
    if retry_after is not None:
        snapshot["retry_after"] = retry_after
    revision = _positive_int(value.get("probe_credential_revision"))
    if revision is not None:
        snapshot["probe_credential_revision"] = revision
    return snapshot


def probe_can_persist_draft(probe: Mapping[str, Any]) -> bool:
    """Return whether a candidate credential may replace its stored predecessor."""

    return str(probe.get("probe_state") or "") in _PERSISTABLE_PROBE_STATES


def probe_snapshot(
    probe: Mapping[str, Any],
    *,
    credential_revision: int | None,
    checked_at: float | None = None,
) -> dict[str, Any]:
    """Build a config-safe snapshot from a connectivity fact."""

    raw = {
        "probe_state": probe.get("probe_state"),
        "credential_state": probe.get("credential_state"),
        "availability": probe.get("availability"),
        "checked_at": checked_at if checked_at is not None else time.time(),
        "retry_after": probe.get("retry_after"),
        "probe_credential_revision": credential_revision,
        "reason_codes": probe.get("reason_codes") or [],
    }
    return sanitize_probe_snapshot(raw)


def derive_provider_state(
    provider: SearchProviderId,
    *,
    configured: bool,
    credential_source: str,
    saved: Mapping[str, Any] | None,
    volatile_probe: Mapping[str, Any] | None = None,
    now: float | None = None,
) -> SearchProviderState:
    """Derive the public state, including staleness and cooldown transitions."""

    current = time.time() if now is None else float(now)
    item = saved if isinstance(saved, Mapping) else {}
    source = credential_source if credential_source in {"config", "secure", "env"} else "none"
    if not configured:
        source = "none"
    enabled = bool(item.get("enabled", True))
    credential_revision = _positive_int(item.get("credential_revision")) if source == "config" else None
    persisted_probe = sanitize_probe_snapshot(item.get("last_probe"))
    probe = sanitize_probe_snapshot(volatile_probe) if source in {"env", "secure"} else persisted_probe

    if not configured:
        probe_state = "not_configured"
        credential_state = "unknown"
        availability = "unavailable"
        checked_at = stale_at = retry_after = None
        probe_revision = None
        reasons = ["provider_not_configured"]
    elif not probe or (
        source == "config"
        and credential_revision is not None
        and _positive_int(probe.get("probe_credential_revision")) != credential_revision
    ):
        probe_state = "not_tested"
        credential_state = "unknown"
        availability = "unknown"
        checked_at = stale_at = retry_after = None
        probe_revision = None
        reasons = ["provider_probe_not_tested"]
    else:
        probe_state = str(probe.get("probe_state") or "not_tested")
        credential_state = str(probe.get("credential_state") or "unknown")
        availability = str(probe.get("availability") or "unknown")
        checked_at = _nonnegative_float(probe.get("checked_at"))
        retry_after = _nonnegative_float(probe.get("retry_after"))
        probe_revision = _positive_int(probe.get("probe_credential_revision"))
        reasons = [str(item) for item in (probe.get("reason_codes") or []) if str(item).strip()]
        stale_at = checked_at + PROBE_STALE_AFTER_S if checked_at is not None and probe_state == "available" else None
        if probe_state == "available" and stale_at is not None and current >= stale_at:
            probe_state = "stale"
            availability = "unknown"
            reasons.append("provider_probe_stale")
        elif probe_state in {"service_unavailable", "network_unreachable"}:
            cooldown_until = (checked_at or current) + SOFT_FAILURE_COOLDOWN_S
            retry_after = retry_after or cooldown_until
            if current >= cooldown_until:
                probe_state = "stale"
                availability = "unknown"
                reasons.append("provider_soft_failure_cooldown_elapsed")
        elif probe_state == "rate_limited":
            retry_after = retry_after or ((checked_at or current) + SOFT_FAILURE_COOLDOWN_S)
            if current >= retry_after:
                probe_state = "stale"
                availability = "unknown"
                reasons.append("provider_rate_limit_elapsed")

    runtime_eligible = enabled and configured and probe_state in _RUNNABLE_PROBE_STATES
    selectable = runtime_eligible
    if not enabled:
        lamp = "gray"
        reasons.append("provider_disabled")
    elif not configured:
        lamp = "gray"
    elif probe_state == "available":
        lamp = "green"
    elif probe_state == "credential_rejected":
        lamp = "red"
    else:
        lamp = "yellow"
    return SearchProviderState(
        provider=provider,
        configured=configured,
        credential_source=source,  # type: ignore[arg-type]
        credential_revision=credential_revision,
        enabled=enabled,
        probe_state=probe_state,
        credential_state=credential_state,
        availability=availability,
        lamp=lamp,
        checked_at=checked_at,
        stale_at=stale_at,
        retry_after=retry_after,
        probe_credential_revision=probe_revision,
        runtime_eligible=runtime_eligible,
        selectable=selectable,
        reason_codes=tuple(dict.fromkeys(reasons)),
    )


def external_search_service_status(states: Mapping[str, Mapping[str, Any] | SearchProviderState]) -> dict[str, Any]:
    """Summarize live External search only; internal lanes are intentionally absent."""

    values = [
        value.to_dict() if isinstance(value, SearchProviderState) else dict(value)
        for provider in SEARCH_PROVIDER_IDS
        if (value := states.get(provider)) is not None
    ]
    enabled_configured = [item for item in values if item.get("enabled") and item.get("configured")]
    runnable = [item for item in enabled_configured if item.get("runtime_eligible")]
    if runnable and all(item.get("probe_state") == "available" for item in enabled_configured):
        status = "ready"
        severity = "info"
        code = "external_search_ready"
        message = "External search providers are available."
    elif runnable:
        status = "warn"
        severity = "warning"
        code = "external_search_degraded"
        message = "External search is usable with warnings or unverified provider state."
    else:
        status = "unavailable"
        severity = "warning"
        code = "external_search_unavailable"
        message = (
            "External search is unavailable. Runs may still use local, Memory, cache, "
            "and indexed_external evidence; grounding remains authoritative."
        )
    return {
        "status": status,
        "runtime_ready": bool(runnable),
        "provider_ready_count": len(runnable),
        "banner": {"code": code, "severity": severity, "message": message},
    }


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _nonnegative_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None
