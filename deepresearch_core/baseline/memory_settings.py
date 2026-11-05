"""Memory refresh settings and structured-output probe metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

JsonDict = dict[str, Any]

WorkspaceMemoryRefreshStrategy = Literal["incremental", "full"]
MEMORY_REFRESH_STRATEGIES: tuple[str, ...] = ("incremental", "full")


@dataclass(frozen=True)
class MemorySettings:
    """User-configurable Memory continuity settings."""

    post_turn_memory_refresh_enabled: bool = True
    workspace_memory_refresh_strategy: WorkspaceMemoryRefreshStrategy = "incremental"
    memory_thinking_enabled: bool = False
    format_probe: JsonDict | None = None

    def to_dict(self) -> JsonDict:
        """Return a JSON-serializable settings payload."""

        return {
            "post_turn_memory_refresh_enabled": self.post_turn_memory_refresh_enabled,
            "workspace_memory_refresh_strategy": self.workspace_memory_refresh_strategy,
            "memory_thinking_enabled": self.memory_thinking_enabled,
            "format_probe": dict(self.format_probe) if isinstance(self.format_probe, dict) else None,
        }


def default_memory_settings() -> MemorySettings:
    """Return default Memory continuity settings."""

    return MemorySettings()


def coerce_memory_settings(data: Any) -> MemorySettings:
    """Validate and normalize a Memory settings payload."""

    if data is None:
        return default_memory_settings()
    if not isinstance(data, dict):
        raise ValueError("memory settings must be an object")
    enabled = data.get("post_turn_memory_refresh_enabled", True)
    if not isinstance(enabled, bool):
        raise ValueError("post_turn_memory_refresh_enabled must be a boolean")
    thinking_enabled = data.get("memory_thinking_enabled", False)
    if not isinstance(thinking_enabled, bool):
        raise ValueError("memory_thinking_enabled must be a boolean")
    strategy = str(data.get("workspace_memory_refresh_strategy") or "incremental").strip().lower()
    if strategy not in MEMORY_REFRESH_STRATEGIES:
        raise ValueError("workspace_memory_refresh_strategy must be one of: incremental, full")
    probe = _safe_probe_payload(data.get("format_probe"))
    return MemorySettings(
        post_turn_memory_refresh_enabled=enabled,
        workspace_memory_refresh_strategy=strategy,  # type: ignore[arg-type]
        memory_thinking_enabled=thinking_enabled,
        format_probe=probe,
    )


def with_format_probe(settings: MemorySettings, probe: JsonDict | None) -> MemorySettings:
    """Return settings with sanitized probe metadata attached."""

    return MemorySettings(
        post_turn_memory_refresh_enabled=settings.post_turn_memory_refresh_enabled,
        workspace_memory_refresh_strategy=settings.workspace_memory_refresh_strategy,
        memory_thinking_enabled=settings.memory_thinking_enabled,
        format_probe=_safe_probe_payload(probe),
    )


def with_stale_format_probe(settings: MemorySettings, *, reason: str) -> MemorySettings:
    """Return settings with the cached probe marked stale when one exists."""

    if not isinstance(settings.format_probe, dict):
        return settings
    probe = {**settings.format_probe, "stale": True, "stale_reason": str(reason or "memory_config_changed")}
    return MemorySettings(
        post_turn_memory_refresh_enabled=settings.post_turn_memory_refresh_enabled,
        workspace_memory_refresh_strategy=settings.workspace_memory_refresh_strategy,
        memory_thinking_enabled=settings.memory_thinking_enabled,
        format_probe=_safe_probe_payload(probe),
    )


def _safe_probe_payload(raw: Any) -> JsonDict | None:
    if not isinstance(raw, dict):
        return None
    allowed = {
        "status",
        "provider_ref",
        "capability",
        "structured_strategy",
        "checked_at",
        "error_code",
        "diagnostics",
        "stale",
        "stale_reason",
    }
    out: JsonDict = {}
    for key in allowed:
        value = raw.get(key)
        if value is None:
            continue
        if key == "diagnostics":
            out[key] = _safe_diagnostics(value)
        elif isinstance(value, (str, int, float, bool)):
            out[key] = value
    return out or None


def _safe_diagnostics(raw: Any) -> JsonDict:
    if not isinstance(raw, dict):
        return {}
    allowed = {"response_shape", "finish_reason", "validation_error", "attempts", "response_empty"}
    diagnostics: JsonDict = {}
    for key in allowed:
        value = raw.get(key)
        if key == "attempts" and isinstance(value, list):
            diagnostics[key] = [_safe_attempt(item) for item in value if isinstance(item, dict)][:5]
        elif isinstance(value, (str, int, float, bool)) or value is None:
            diagnostics[key] = value
    return diagnostics


def _safe_attempt(raw: JsonDict) -> JsonDict:
    out: JsonDict = {}
    for key in ("ok", "structured_strategy", "error_code", "response_shape"):
        value = raw.get(key)
        if isinstance(value, (str, int, float, bool)) or value is None:
            out[key] = value
    return out
