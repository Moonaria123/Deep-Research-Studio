"""Persisted user setting for Deep Research automatic checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeepCheckpointSettings:
    """Global Deep checkpoint policy frozen into each new run."""

    auto_checkpoint_enabled: bool = False

    def to_dict(self) -> dict[str, bool]:
        return {"auto_checkpoint_enabled": self.auto_checkpoint_enabled}


def default_deep_checkpoint_settings() -> DeepCheckpointSettings:
    """Return the privacy- and cost-conservative product default."""

    return DeepCheckpointSettings()


def coerce_deep_checkpoint_settings(value: Any) -> DeepCheckpointSettings:
    """Validate the complete Settings API payload."""

    if value is None:
        return default_deep_checkpoint_settings()
    if not isinstance(value, dict):
        raise ValueError("deep checkpoint settings must be an object")
    enabled = value.get("auto_checkpoint_enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("auto_checkpoint_enabled must be a boolean")
    return DeepCheckpointSettings(auto_checkpoint_enabled=enabled)
