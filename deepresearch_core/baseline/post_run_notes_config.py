"""Configuration for automatic post-run Library notes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .budget import Mode

JsonDict = dict[str, Any]
POST_RUN_NOTE_MODES = tuple(mode.value for mode in Mode)


@dataclass(frozen=True)
class PostRunNotesConfig:
    """Modes that may write deterministic post-run notes."""

    enabled_modes: tuple[str, ...] = ()

    def to_dict(self) -> JsonDict:
        return {"enabled_modes": list(self.enabled_modes)}


def default_post_run_notes_config() -> PostRunNotesConfig:
    """Default to no automatic notes; users opt in per mode."""

    return PostRunNotesConfig()


def coerce_post_run_notes_config(data: Any) -> PostRunNotesConfig:
    """Validate and normalize a post-run notes config payload."""

    if data is None:
        return default_post_run_notes_config()
    if not isinstance(data, dict):
        raise ValueError("post_run_notes config must be an object")
    raw = data.get("enabled_modes", [])
    if raw is None:
        raw = []
    if not isinstance(raw, list):
        raise ValueError("enabled_modes must be a list")
    modes: list[str] = []
    allowed = set(POST_RUN_NOTE_MODES)
    for item in raw:
        mode = str(item or "").strip().lower()
        if mode not in allowed:
            raise ValueError("enabled_modes contains an unknown mode")
        if mode not in modes:
            modes.append(mode)
    return PostRunNotesConfig(enabled_modes=tuple(modes))
