"""Context injection budget config, resolution, and heuristic recommendation (DR-ITER-057).

Three continuity blocks share one character budget, filled by a *fixed* priority
order: recent linear turns -> semantic same-session recall -> long-term memory.
This module owns the user-configurable knobs (named presets + custom shares) and
resolves the effective per-run config from the precedence chain:

    run override  >  saved config.json  >  DEEPRESEARCH_CONTEXT_* env  >  built-in preset

Presets are defined as absolute character allocations (summing to the default
total) so the ``balanced`` preset reproduces the original T5 caps exactly; custom
configs come in as integer shares (0..100) and are converted to caps with the
*remainder added to the recent block* (the highest-priority block), guaranteeing
the three caps sum to ``max_chars``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# --- defaults + named presets -------------------------------------------------

DEFAULT_MAX_CHARS = 6000
DEFAULT_TURNS = 6
SHARE_TOLERANCE = 1  # accept 99..101 so a slider UI never trips on rounding

PRESET_DEFAULT = "balanced"
PRESET_CUSTOM = "custom"

# Char allocations at the default total (recent, semantic, memory); each sums to
# ``DEFAULT_MAX_CHARS``. ``balanced`` mirrors the DR-ITER-053/054 env defaults.
PRESET_CHARS: dict[str, tuple[int, int, int]] = {
    "balanced": (3500, 1500, 1000),     # ~58 / 25 / 17
    "followup": (4200, 1200, 600),      # 70 / 20 / 10 -- favors the live thread
    "long_thread": (3000, 2100, 900),   # 50 / 35 / 15 -- favors semantic recall
    "memory_rich": (2700, 900, 2400),   # 45 / 15 / 40 -- favors long-term memory
}
PRESETS = tuple(PRESET_CHARS.keys())

# Env vars (deploy-level fallback); absolute character caps, matching orchestrator.
ENV_MAX_CHARS = "DEEPRESEARCH_CONTEXT_MAX_CHARS"
ENV_TURNS = "DEEPRESEARCH_CONTEXT_TURNS"
ENV_RECENT = "DEEPRESEARCH_CONTEXT_RECENT_CHARS"
ENV_SEMANTIC = "DEEPRESEARCH_CONTEXT_SEMANTIC_CHARS"
ENV_MEMORY = "DEEPRESEARCH_CONTEXT_MEMORY_CHARS"
_ENV_CONTEXT_VARS = (ENV_MAX_CHARS, ENV_TURNS, ENV_RECENT, ENV_SEMANTIC, ENV_MEMORY)


@dataclass(frozen=True)
class ContextBudgetConfig:
    """Effective context budget: a fixed-order, char-canonical allocation.

    ``recent_chars`` / ``semantic_chars`` / ``memory_chars`` are the source of
    truth (consumed by the orchestrator). ``preset`` is a display label and
    ``shares()`` derives integer percentages for the UI.
    """

    preset: str = PRESET_DEFAULT
    max_chars: int = DEFAULT_MAX_CHARS
    recent_chars: int = 3500
    semantic_chars: int = 1500
    memory_chars: int = 1000
    turns: int = DEFAULT_TURNS

    def caps(self) -> tuple[int, int, int]:
        """Return the (recent, semantic, memory) character caps."""

        return (self.recent_chars, self.semantic_chars, self.memory_chars)

    def shares(self) -> tuple[int, int, int]:
        """Return integer (recent, semantic, memory) percentages for display."""

        total = self.max_chars or (self.recent_chars + self.semantic_chars + self.memory_chars)
        if total <= 0:
            return (0, 0, 0)
        semantic = round(self.semantic_chars / total * 100)
        memory = round(self.memory_chars / total * 100)
        recent = 100 - semantic - memory
        return (max(0, recent), semantic, memory)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view (caps + derived shares for the UI)."""

        recent_share, semantic_share, memory_share = self.shares()
        return {
            "preset": self.preset,
            "max_chars": self.max_chars,
            "turns": self.turns,
            "recent_chars": self.recent_chars,
            "semantic_chars": self.semantic_chars,
            "memory_chars": self.memory_chars,
            "recent_share": recent_share,
            "semantic_share": semantic_share,
            "memory_share": memory_share,
        }


# --- helpers ------------------------------------------------------------------


def _pos_int(value: Any, default: int) -> int:
    """Coerce to a positive int, falling back to ``default`` on any bad value."""

    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    return out if out > 0 else default


def _nonneg_int(value: Any, default: int = 0) -> int:
    """Coerce to a non-negative int, falling back to ``default``."""

    try:
        out = int(value)
    except (TypeError, ValueError):
        return default
    return out if out >= 0 else default


def _scale_caps(chars: tuple[int, int, int], max_chars: int) -> tuple[int, int, int]:
    """Scale a base (recent, semantic, memory) tuple to ``max_chars``.

    Remainder is added to the recent block so the three caps sum to ``max_chars``.
    """

    base_total = sum(chars) or DEFAULT_MAX_CHARS
    semantic = chars[1] * max_chars // base_total
    memory = chars[2] * max_chars // base_total
    recent = max_chars - semantic - memory
    return (max(0, recent), max(0, semantic), max(0, memory))


def _shares_to_caps(recent_s: int, semantic_s: int, memory_s: int, max_chars: int) -> tuple[int, int, int]:
    """Convert integer shares to caps, adding the rounding remainder to recent."""

    semantic = max_chars * semantic_s // 100
    memory = max_chars * memory_s // 100
    recent = max_chars - semantic - memory
    return (max(0, recent), max(0, semantic), max(0, memory))


def from_preset(preset: str, *, max_chars: int = DEFAULT_MAX_CHARS, turns: int = DEFAULT_TURNS) -> ContextBudgetConfig:
    """Build a config from a named preset, scaled to ``max_chars``."""

    base = PRESET_CHARS[preset]
    recent, semantic, memory = _scale_caps(base, max_chars)
    return ContextBudgetConfig(
        preset=preset, max_chars=max_chars, recent_chars=recent,
        semantic_chars=semantic, memory_chars=memory, turns=turns,
    )


def default_config() -> ContextBudgetConfig:
    """Return the built-in default (``balanced`` at the default total)."""

    return from_preset(PRESET_DEFAULT)


def coerce_payload(payload: Mapping[str, Any]) -> ContextBudgetConfig:
    """Build + validate a config from a user/API payload.

    Accepts three shapes: a named ``preset`` (optionally with ``max_chars`` /
    ``turns``), a ``custom`` config with integer shares, or a fully-resolved dict
    with explicit ``*_chars`` (round-trip from ``to_dict``). Raises ``ValueError``
    on any invalid shape; callers map this to ``INVALID_REQUEST``.
    """

    if not isinstance(payload, Mapping):
        raise ValueError("context_budget must be an object")
    preset = str(payload.get("preset") or "").strip().lower() or None
    max_chars = _pos_int(payload.get("max_chars"), DEFAULT_MAX_CHARS)
    turns = _pos_int(payload.get("turns"), DEFAULT_TURNS)

    # 1) explicit caps (resolved dict round-trip): trust the caps, validate basics.
    if any(k in payload for k in ("recent_chars", "semantic_chars", "memory_chars")):
        recent = _pos_int(payload.get("recent_chars"), 0)
        semantic = _nonneg_int(payload.get("semantic_chars"), 0)
        memory = _nonneg_int(payload.get("memory_chars"), 0)
        if recent <= 0:
            raise ValueError("recent_chars must be a positive integer")
        total = max_chars if "max_chars" in payload else (recent + semantic + memory)
        return ContextBudgetConfig(
            preset=preset or PRESET_CUSTOM, max_chars=total, recent_chars=recent,
            semantic_chars=semantic, memory_chars=memory, turns=turns,
        )

    # 2) named preset.
    if preset and preset in PRESET_CHARS:
        return from_preset(preset, max_chars=max_chars, turns=turns)

    # 3) custom shares.
    share_keys = ("recent_share", "semantic_share", "memory_share")
    if all(payload.get(k) is not None for k in share_keys):
        recent_s = _nonneg_int(payload.get("recent_share"))
        semantic_s = _nonneg_int(payload.get("semantic_share"))
        memory_s = _nonneg_int(payload.get("memory_share"))
        if recent_s <= 0:
            raise ValueError("recent_share must be a positive integer")
        total_share = recent_s + semantic_s + memory_s
        if not (100 - SHARE_TOLERANCE <= total_share <= 100 + SHARE_TOLERANCE):
            raise ValueError("recent/semantic/memory shares must sum to 100")
        recent, semantic, memory = _shares_to_caps(recent_s, semantic_s, memory_s, max_chars)
        return ContextBudgetConfig(
            preset=PRESET_CUSTOM, max_chars=max_chars, recent_chars=recent,
            semantic_chars=semantic, memory_chars=memory, turns=turns,
        )

    if preset is not None:
        raise ValueError(f"unknown context_budget preset '{preset}'")
    raise ValueError("context_budget requires a preset or recent/semantic/memory shares")


def from_env(env: Mapping[str, str]) -> ContextBudgetConfig | None:
    """Build a config from ``DEEPRESEARCH_CONTEXT_*`` env, or ``None`` when unset.

    Only the absolute per-block char caps are read; the preset is labeled
    ``custom`` since env is a free-form deploy override.
    """

    if not any(str(env.get(name) or "").strip() for name in _ENV_CONTEXT_VARS):
        return None
    recent = _pos_int(env.get(ENV_RECENT), 3500)
    semantic = _nonneg_int(env.get(ENV_SEMANTIC), 1500)
    memory = _nonneg_int(env.get(ENV_MEMORY), 1000)
    max_chars = _pos_int(env.get(ENV_MAX_CHARS), recent + semantic + memory)
    turns = _pos_int(env.get(ENV_TURNS), DEFAULT_TURNS)
    return ContextBudgetConfig(
        preset=PRESET_CUSTOM, max_chars=max_chars, recent_chars=recent,
        semantic_chars=semantic, memory_chars=memory, turns=turns,
    )


def resolve_context_budget(
    run_override: Mapping[str, Any] | None,
    saved: ContextBudgetConfig | None,
    env: Mapping[str, str],
) -> ContextBudgetConfig:
    """Resolve the effective config along the precedence chain.

    run override (Composer single-run) > saved config.json (Settings) >
    DEEPRESEARCH_CONTEXT_* env (deploy) > built-in ``balanced`` preset.
    """

    if run_override:
        return coerce_payload(run_override)
    if saved is not None:
        return saved
    env_config = from_env(env)
    if env_config is not None:
        return env_config
    return default_config()


# --- heuristic recommendation (no LLM) ----------------------------------------


def recommend(
    *,
    embed_ready: bool,
    workspace_memory_count: int,
    session_turn_count: int,
    memory_policy: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    turns: int = DEFAULT_TURNS,
) -> dict[str, Any]:
    """Recommend a preset from runtime signals, with human-readable reasons.

    Returns ``{"recommendation": <config dict>, "reasons": [...]}``. Pure
    heuristic: no LLM call. When semantic recall is impossible (no embedder) the
    semantic block is zeroed and its share is folded into the recent block.
    """

    reasons: list[str] = []
    memory_active = memory_policy and memory_policy != "off"

    if not embed_ready:
        config = from_preset("followup", max_chars=max_chars, turns=turns)
        # No embedder -> semantic recall returns nothing; fold its budget into recent.
        config = ContextBudgetConfig(
            preset="custom", max_chars=config.max_chars,
            recent_chars=config.recent_chars + config.semantic_chars,
            semantic_chars=0, memory_chars=config.memory_chars, turns=config.turns,
        )
        reasons.append(
            "No embedding model configured: semantic recall is unavailable; "
            "related earlier turns quota is zeroed and rolled into recent conversation."
        )
        if memory_active and workspace_memory_count >= 5:
            reasons.append(
                f"Workspace has {workspace_memory_count} long-term memories; keeping memory quota."
            )
        return {"recommendation": config.to_dict(), "reasons": reasons}

    if memory_active and workspace_memory_count >= 5:
        reasons.append(
            f"Workspace has {workspace_memory_count} long-term memories with memory injection on: "
            "favor long-term memory."
        )
        config = from_preset("memory_rich", max_chars=max_chars, turns=turns)
        return {"recommendation": config.to_dict(), "reasons": reasons}

    if session_turn_count > 12:
        reasons.append(
            f"This session has {session_turn_count} messages: favor semantic recall to connect earlier content."
        )
        config = from_preset("long_thread", max_chars=max_chars, turns=turns)
        return {"recommendation": config.to_dict(), "reasons": reasons}

    reasons.append("Session size and memory volume are moderate: using the balanced preset.")
    if not memory_active:
        reasons.append(
            "Long-term memory injection is off: the memory block only applies when workspace memory is enabled."
        )
    config = from_preset("balanced", max_chars=max_chars, turns=turns)
    return {"recommendation": config.to_dict(), "reasons": reasons}
