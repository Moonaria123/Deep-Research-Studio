"""LLM provider registry: multi-provider, multi-model BYOK configuration.

The registry holds named :class:`LLMProviderProfile` entries plus role bindings
(default LLM, embedder, council members). Each binding is a ``"<provider_id>:<model>"``
reference. All public CRUD returns a new immutable registry (no in-place mutation),
and :meth:`LLMRegistry.status` is strictly secret-free (raw keys never leak).

Search tools (Brave/Tavily/Exa/Firecrawl) remain in :class:`ProviderSettings`; this module is
LLM-only and lives alongside them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .settings import _configured_value

# Provider type tags understood by the resolver. ``ollama`` maps to the local
# Ollama adapter; everything else is treated as an OpenAI-compatible endpoint.
PROVIDER_TYPES = ("openai", "ollama", "anthropic", "litellm")

# Model kinds: ``chat`` (reasoning/answering) vs ``embed`` (vector embeddings).
# The embed kind is the single source for all vector indexing (documents, library,
# memory, conversation history). Multimodal vs text-only is a sub-distinction of
# ``chat`` expressed via ``ModelCapabilities.vision``/``ocr``.
MODEL_KINDS = ("chat", "embed")

_DEFAULT_CONTEXT_WINDOW = 128000


def _int_or(value: Any, default: int) -> int:
    """Return ``value`` parsed as a positive int, or ``default``."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _int_or_none(value: Any) -> int | None:
    """Return ``value`` parsed as a positive int, or ``None`` when absent/invalid."""

    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _temperature_or_none(value: Any) -> float | None:
    """Return ``value`` parsed as a sampling temperature in ``[0, 2]``, else ``None``."""

    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(2.0, parsed))


@dataclass(frozen=True)
class ModelCapabilities:
    """Per-model capability flags used as routing/assignment signals (trimmed set)."""

    vision: bool = False
    ocr: bool = False
    web_search: bool = False
    thinking: bool = False

    def to_config(self) -> dict[str, bool]:
        """Return the persistable shape."""

        return {
            "vision": bool(self.vision),
            "ocr": bool(self.ocr),
            "web_search": bool(self.web_search),
            "thinking": bool(self.thinking),
        }

    @classmethod
    def from_config(cls, data: Any) -> "ModelCapabilities":
        """Build capabilities from a dict (missing keys default to False)."""

        if not isinstance(data, dict):
            return cls()
        return cls(
            vision=bool(data.get("vision", False)),
            ocr=bool(data.get("ocr", False)),
            web_search=bool(data.get("web_search", False)),
            thinking=bool(data.get("thinking", False)),
        )


@dataclass(frozen=True)
class ModelProfile:
    """A single model entry under a provider, with category + trimmed capabilities."""

    id: str
    alias: str = ""
    kind: str = "chat"
    context_window: int = _DEFAULT_CONTEXT_WINDOW
    capabilities: ModelCapabilities = field(default_factory=ModelCapabilities)
    thinking_budget: int | None = None
    temperature: float | None = None
    dimensions: int | None = None
    enabled: bool = True

    @property
    def is_embed(self) -> bool:
        """Return whether this is a vector-embedding model."""

        return self.kind == "embed"

    @property
    def is_multimodal(self) -> bool:
        """Return whether this chat model handles non-text input (vision/ocr)."""

        return bool(self.capabilities.vision or self.capabilities.ocr)

    def display_name(self) -> str:
        """Return the alias when set, else the raw model id."""

        return self.alias or self.id

    def to_config(self) -> dict[str, Any]:
        """Return the persistable shape (no secrets exist on a model profile)."""

        data: dict[str, Any] = {
            "id": self.id,
            "alias": self.alias,
            "kind": self.kind,
            "context_window": self.context_window,
            "capabilities": self.capabilities.to_config(),
            "enabled": bool(self.enabled),
        }
        if self.thinking_budget is not None:
            data["thinking_budget"] = self.thinking_budget
        if self.temperature is not None:
            data["temperature"] = self.temperature
        if self.dimensions is not None:
            data["dimensions"] = self.dimensions
        return data

    def status(self) -> dict[str, Any]:
        """Return a secret-free view (identical to ``to_config`` -- no secrets)."""

        return self.to_config()

    @classmethod
    def from_config(cls, data: Any) -> "ModelProfile":
        """Build a model profile from a str (legacy), dict, or existing ModelProfile."""

        if isinstance(data, ModelProfile):
            return data
        if isinstance(data, str):
            return cls(id=data.strip())
        if not isinstance(data, dict):
            return cls(id=str(data or "").strip())
        kind = str(data.get("kind") or "chat").strip().lower()
        if kind not in MODEL_KINDS:
            kind = "chat"
        return cls(
            id=str(data.get("id") or "").strip(),
            alias=str(data.get("alias") or "").strip(),
            kind=kind,
            context_window=_int_or(data.get("context_window"), _DEFAULT_CONTEXT_WINDOW),
            capabilities=ModelCapabilities.from_config(data.get("capabilities")),
            thinking_budget=_int_or_none(data.get("thinking_budget")),
            temperature=_temperature_or_none(data.get("temperature")),
            dimensions=_int_or_none(data.get("dimensions")),
            enabled=bool(data.get("enabled", True)),
        )


def _model_has_id(model: Any) -> bool:
    """Return whether a raw model entry (str/dict/ModelProfile) carries a non-empty id."""

    if isinstance(model, ModelProfile):
        return bool(model.id)
    if isinstance(model, str):
        return bool(model.strip())
    if isinstance(model, dict):
        return bool(str(model.get("id") or "").strip())
    return False


@dataclass(frozen=True)
class LLMProviderProfile:
    """A single named LLM provider (one base_url + key + known model list).

    ``models`` accepts legacy ``str`` entries, dicts, or :class:`ModelProfile`
    objects; all are coerced to :class:`ModelProfile` (empty ids dropped) so old
    string-list configs keep working transparently.
    """

    id: str
    label: str = ""
    type: str = "openai"
    base_url: str = ""
    api_key: str = ""
    models: tuple[ModelProfile, ...] = ()
    enabled: bool = True

    def __post_init__(self) -> None:
        coerced = tuple(ModelProfile.from_config(m) for m in self.models if _model_has_id(m))
        object.__setattr__(self, "models", coerced)

    def has_key(self) -> bool:
        """Return whether a real (non-placeholder) API key is configured."""

        return _configured_value(self.api_key)

    def model_ids(self) -> tuple[str, ...]:
        """Return the bare model ids (order-preserving)."""

        return tuple(m.id for m in self.models)

    def get_model(self, model_id: str) -> ModelProfile | None:
        """Return the :class:`ModelProfile` matching ``model_id`` or ``None``."""

        key = str(model_id or "").strip()
        for model in self.models:
            if model.id == key:
                return model
        return None

    def status(self) -> dict[str, Any]:
        """Return a secret-free view (no raw api_key)."""

        return {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "base_url": self.base_url,
            "models": [m.status() for m in self.models],
            "enabled": bool(self.enabled),
            "has_key": self.has_key(),
        }

    def to_config(self) -> dict[str, Any]:
        """Return the full persistable shape (includes api_key for local config)."""

        return {
            "id": self.id,
            "label": self.label,
            "type": self.type,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "models": [m.to_config() for m in self.models],
            "enabled": bool(self.enabled),
        }

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> "LLMProviderProfile":
        """Build a profile from a persisted/parsed dict."""

        models = data.get("models")
        model_tuple = tuple(models) if isinstance(models, list) else ()
        return cls(
            id=str(data.get("id") or "").strip(),
            label=str(data.get("label") or "").strip(),
            type=str(data.get("type") or "openai").strip() or "openai",
            base_url=str(data.get("base_url") or "").strip(),
            api_key=str(data.get("api_key") or ""),
            models=model_tuple,
            enabled=bool(data.get("enabled", True)),
        )


@dataclass(frozen=True)
class LLMRoles:
    """Role bindings: which ``provider:model`` plays each role."""

    default: str = ""
    embed: str = ""
    memory: str = ""
    council: tuple[str, ...] = ()

    def to_config(self) -> dict[str, Any]:
        """Return the persistable shape."""

        return {"default": self.default, "embed": self.embed, "memory": self.memory, "council": list(self.council)}

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> "LLMRoles":
        """Build roles from a persisted/parsed dict."""

        council = data.get("council")
        council_tuple = tuple(str(c).strip() for c in council if str(c).strip()) if isinstance(council, list) else ()
        return cls(
            default=str(data.get("default") or "").strip(),
            embed=str(data.get("embed") or "").strip(),
            memory=str(data.get("memory") or "").strip(),
            council=council_tuple,
        )


def _provider_id_of(ref: str) -> str:
    """Return the provider id portion of a ``provider:model`` reference."""

    text = str(ref or "").strip()
    return text.split(":", 1)[0] if text else ""


@dataclass(frozen=True)
class LLMRegistry:
    """Immutable collection of provider profiles plus role bindings."""

    providers: tuple[LLMProviderProfile, ...] = ()
    roles: LLMRoles = field(default_factory=LLMRoles)

    # --- lookup ---------------------------------------------------------------

    def get(self, provider_id: str) -> LLMProviderProfile | None:
        """Return the profile with ``provider_id`` or ``None``."""

        key = str(provider_id or "").strip()
        for profile in self.providers:
            if profile.id == key:
                return profile
        return None

    def resolve(self, ref: str) -> tuple[LLMProviderProfile, str] | None:
        """Resolve a ``provider:model`` reference into ``(profile, model)``.

        Splits on the first colon only, so model names containing colons
        (e.g. ``llama3.1:8b``) are preserved.
        """

        text = str(ref or "").strip()
        if not text or ":" not in text:
            # A bare provider id with no model is not a usable binding.
            profile = self.get(text)
            return None if profile is None else (profile, "")
        provider_id, model = text.split(":", 1)
        profile = self.get(provider_id)
        if profile is None:
            return None
        return profile, model.strip()

    # --- CRUD (return new registries) ----------------------------------------

    def upsert(self, profile: LLMProviderProfile) -> "LLMRegistry":
        """Add or replace a profile by id; returns a new registry."""

        if not str(profile.id or "").strip():
            raise ValueError("provider id is required")
        replaced = False
        new_providers: list[LLMProviderProfile] = []
        for existing in self.providers:
            if existing.id == profile.id:
                new_providers.append(profile)
                replaced = True
            else:
                new_providers.append(existing)
        if not replaced:
            new_providers.append(profile)
        return replace(self, providers=tuple(new_providers))

    def remove(self, provider_id: str) -> "LLMRegistry":
        """Remove a profile and scrub any role bindings that referenced it."""

        key = str(provider_id or "").strip()
        new_providers = tuple(p for p in self.providers if p.id != key)
        roles = self.roles
        default = "" if _provider_id_of(roles.default) == key else roles.default
        embed = "" if _provider_id_of(roles.embed) == key else roles.embed
        memory = "" if _provider_id_of(roles.memory) == key else roles.memory
        council = tuple(ref for ref in roles.council if _provider_id_of(ref) != key)
        return replace(self, providers=new_providers, roles=LLMRoles(default=default, embed=embed, memory=memory, council=council))

    def upsert_model(self, provider_id: str, model: ModelProfile) -> "LLMRegistry":
        """Add or replace a single model under a provider; returns a new registry."""

        profile = self.get(provider_id)
        if profile is None:
            raise ValueError(f"unknown provider: {provider_id}")
        if not str(model.id or "").strip():
            raise ValueError("model id is required")
        replaced = False
        new_models: list[ModelProfile] = []
        for existing in profile.models:
            if existing.id == model.id:
                new_models.append(model)
                replaced = True
            else:
                new_models.append(existing)
        if not replaced:
            new_models.append(model)
        return self.upsert(replace(profile, models=tuple(new_models)))

    def remove_model(self, provider_id: str, model_id: str) -> "LLMRegistry":
        """Remove a model from a provider, scrubbing any ``provider:model`` role refs."""

        profile = self.get(provider_id)
        if profile is None:
            raise ValueError(f"unknown provider: {provider_id}")
        key = str(model_id or "").strip()
        new_models = tuple(m for m in profile.models if m.id != key)
        registry = self.upsert(replace(profile, models=new_models))
        ref = f"{provider_id}:{key}"
        roles = registry.roles
        default = "" if roles.default == ref else roles.default
        embed = "" if roles.embed == ref else roles.embed
        memory = "" if roles.memory == ref else roles.memory
        council = tuple(r for r in roles.council if r != ref)
        return replace(registry, roles=LLMRoles(default=default, embed=embed, memory=memory, council=council))

    def set_roles(
        self,
        *,
        default: str | None = None,
        embed: str | None = None,
        memory: str | None = None,
        council: tuple[str, ...] | list[str] | None = None,
    ) -> "LLMRegistry":
        """Set role bindings, validating each reference points to a known provider."""

        roles = self.roles
        new_default = roles.default if default is None else str(default or "").strip()
        new_embed = roles.embed if embed is None else str(embed or "").strip()
        new_memory = roles.memory if memory is None else str(memory or "").strip()
        new_council = (
            roles.council
            if council is None
            else tuple(str(c).strip() for c in council if str(c).strip())
        )
        for ref in [new_default, new_embed, new_memory, *new_council]:
            if ref and self.get(_provider_id_of(ref)) is None:
                raise ValueError(f"role binding references unknown provider: {ref}")
        return replace(self, roles=LLMRoles(default=new_default, embed=new_embed, memory=new_memory, council=new_council))

    # --- serialization --------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Return a strictly secret-free registry snapshot."""

        return {
            "providers": [p.status() for p in self.providers],
            "roles": self.roles.to_config(),
        }

    def to_config(self) -> dict[str, Any]:
        """Return the persistable shape (includes api_keys for local config only)."""

        return {
            "llm_providers": [p.to_config() for p in self.providers],
            "llm_roles": self.roles.to_config(),
        }

    @classmethod
    def from_config(cls, data: dict[str, Any]) -> "LLMRegistry":
        """Build a registry from persisted ``llm_providers`` + ``llm_roles``."""

        raw_providers = data.get("llm_providers")
        providers = (
            tuple(LLMProviderProfile.from_config(item) for item in raw_providers if isinstance(item, dict))
            if isinstance(raw_providers, list)
            else ()
        )
        raw_roles = data.get("llm_roles")
        roles = LLMRoles.from_config(raw_roles) if isinstance(raw_roles, dict) else LLMRoles()
        return cls(providers=providers, roles=roles)
