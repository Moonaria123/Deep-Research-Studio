import json
import os
import threading
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .env import provider_settings_from_mapping
from .providers_registry import LLMProviderProfile, LLMRegistry, LLMRoles, ModelProfile
from .secret_vault import protect_config_secrets, resolve_config_secrets
from .settings import ProviderSettings, _configured_value


PROVIDER_FIELDS = (
    "llm_base_url",
    "llm_api_key",
    "llm_model",
    "brave_api_key",
    "firecrawl_api_key",
    "firecrawl_base_url",
    "exa_api_key",
    "tavily_api_key",
)
PROVIDER_ENV_VARS = (
    "DEEPRESEARCH_LLM_BASE_URL",
    "DEEPRESEARCH_LLM_API_KEY",
    "DEEPRESEARCH_LLM_MODEL",
    "DEEPRESEARCH_BRAVE_API_KEY",
    "DEEPRESEARCH_FIRECRAWL_API_KEY",
    "DEEPRESEARCH_FIRECRAWL_BASE_URL",
    "DEEPRESEARCH_EXA_API_KEY",
    "DEEPRESEARCH_TAVILY_API_KEY",
)
ASSET_ENV_VARS = ("DEEPRESEARCH_REPORTS_DIR", "DEEPRESEARCH_HISTORY_DIR")
MEMORY_DREAM_SCHEDULES = ("off", "daily", "weekly")
_CONFIG_LOCKS: dict[str, threading.RLock] = {}
_CONFIG_LOCKS_GUARD = threading.Lock()


class SearchProviderStateConflict(RuntimeError):
    """Raised when a search-provider CAS observes a newer config revision."""

    def __init__(self, *, expected: int, actual: int) -> None:
        super().__init__(f"search provider state revision changed: expected {expected}, got {actual}")
        self.expected = expected
        self.actual = actual


def default_config_path(environ: Mapping[str, str] | None = None) -> Path:
    """Return the default user-local CLI configuration path."""

    env = environ if environ is not None else os.environ
    explicit = str(env.get("DEEPRESEARCH_CONFIG") or "").strip()
    if explicit:
        return Path(explicit).expanduser()
    appdata = str(env.get("APPDATA") or "").strip()
    if appdata:
        return Path(appdata).expanduser() / "DeepResearch" / "config.json"
    xdg_config = str(env.get("XDG_CONFIG_HOME") or "").strip()
    if xdg_config:
        return Path(xdg_config).expanduser() / "deepresearch" / "config.json"
    return Path.home() / ".config" / "deepresearch" / "config.json"


def load_provider_config(path: Path, *, read_only: bool = False) -> ProviderSettings:
    """Load provider settings from a user-local JSON config file."""

    data = _read_config_dict(path, read_only=read_only)
    provider = data.get("provider", data)
    if not isinstance(provider, dict):
        raise ValueError(f"Invalid DeepResearch provider config: {path}")
    values = {field: _string_value(provider.get(field)) for field in PROVIDER_FIELDS}
    return ProviderSettings(**values)


def save_provider_config(path: Path, settings: ProviderSettings) -> None:
    """Save provider settings to a user-local JSON config file.

    Replaces only the ``provider`` block; other top-level keys (e.g. the LLM
    registry ``llm_providers``/``llm_roles``) are preserved.
    """

    provider_block = {
        field: getattr(settings, field)
        for field in PROVIDER_FIELDS
        if str(getattr(settings, field) or "").strip()
    }
    _update_config_dict(path, lambda data: data.__setitem__("provider", provider_block))


def save_non_search_provider_config(path: Path, settings: ProviderSettings) -> None:
    """Persist only LLM fields while preserving concurrent search-provider state."""

    search_fields = {
        "brave_api_key",
        "tavily_api_key",
        "exa_api_key",
        "firecrawl_api_key",
        "firecrawl_base_url",
    }

    def mutate(data: dict[str, Any]) -> None:
        raw_provider = data.get("provider", data)
        provider_block = (
            {
                field: raw_provider[field]
                for field in PROVIDER_FIELDS
                if isinstance(raw_provider, Mapping)
                and field in raw_provider
                and str(raw_provider[field] or "").strip()
            }
            if isinstance(raw_provider, Mapping)
            else {}
        )
        for field in PROVIDER_FIELDS:
            if field in search_fields:
                continue
            value = str(getattr(settings, field) or "").strip()
            if value:
                provider_block[field] = value
            else:
                provider_block.pop(field, None)
        data["provider"] = provider_block

    _update_config_dict(path, mutate)


def load_search_usage_config(path: Path, *, read_only: bool = False):
    """Load the saved search usage policy, defaulting safely for old configs."""

    from .baseline.search_usage import DEFAULT_SEARCH_USAGE_POLICY, normalize_search_usage_policy

    data = _read_config_dict(path, read_only=read_only)
    block = data.get("search_usage")
    if block is None:
        return DEFAULT_SEARCH_USAGE_POLICY
    try:
        return normalize_search_usage_policy(block, require_complete=True)
    except ValueError as error:
        raise ValueError(f"Invalid search_usage config in {path}") from error


def save_search_usage_config(path: Path, policy) -> None:
    """Save search usage separately while preserving provider and registry blocks."""

    from .baseline.search_usage import normalize_search_usage_policy

    normalized = normalize_search_usage_policy(policy, require_complete=True)
    _update_config_dict(
        path,
        lambda data: data.__setitem__("search_usage", normalized.to_dict()),
    )


def load_search_provider_state_config(path: Path, *, read_only: bool = False) -> dict[str, Any]:
    """Load the normalized External-search state block without writing legacy files."""

    from .baseline.search_provider_state import normalize_state_block

    data = _read_config_dict(path, read_only=read_only)
    return normalize_state_block(data.get("search_provider_state"))


def commit_search_provider_config(
    path: Path,
    *,
    expected_revision: int,
    provider_updates: Mapping[str, str | None],
    provider_states: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """CAS-update search credentials and their secret-free state in one file write.

    Only search-provider fields named by ``provider_updates`` are touched. This
    prevents an effective env/secure overlay from being copied into plaintext and
    preserves unrelated provider, registry, retrieval, and Memory configuration.
    """

    from .baseline.search_provider_state import SEARCH_PROVIDER_IDS, normalize_state_block

    with _config_write_lock(path):
        data, persisted = _read_config_pair_unlocked(path)
        previous = json.loads(json.dumps(data))
        current = normalize_state_block(data.get("search_provider_state"))
        actual = int(current.get("revision") or 0)
        if actual != int(expected_revision):
            raise SearchProviderStateConflict(expected=int(expected_revision), actual=actual)

        raw_provider = data.get("provider", data)
        provider_block = (
            {
                field: raw_provider[field]
                for field in PROVIDER_FIELDS
                if field in raw_provider and str(raw_provider[field] or "").strip()
            }
            if isinstance(raw_provider, Mapping)
            else {}
        )
        allowed_fields = {
            "brave_api_key",
            "tavily_api_key",
            "exa_api_key",
            "firecrawl_api_key",
            "firecrawl_base_url",
        }
        for field, value in provider_updates.items():
            if field not in allowed_fields:
                raise ValueError(f"unsupported search provider config field: {field}")
            text = str(value or "").strip()
            if value is None or not text:
                provider_block.pop(field, None)
            else:
                provider_block[field] = text

        normalized_providers: dict[str, dict[str, Any]] = {}
        for provider in SEARCH_PROVIDER_IDS:
            item = provider_states.get(provider)
            if not isinstance(item, Mapping):
                item = current["providers"][provider]
            normalized_providers[provider] = dict(item)
        next_state = normalize_state_block(
            {
                "revision": actual + 1,
                "providers": normalized_providers,
            }
        )
        data["provider"] = provider_block
        data["search_provider_state"] = next_state
        _write_config_dict_unlocked(
            path,
            data,
            previous_plain=previous,
            previous_protected=persisted,
        )
        return next_state


def _read_config_dict(path: Path, *, read_only: bool = False) -> dict[str, Any]:
    """Return decrypted runtime config, migrating legacy raw secrets atomically."""

    if read_only:
        raw = _read_raw_config_dict(path)
        runtime, _protected, _migrated = resolve_config_secrets(
            path,
            raw,
            migrate=False,
            read_only=True,
        )
        if not isinstance(runtime, dict):
            raise ValueError(f"Invalid DeepResearch config shape: {path}")
        return runtime

    with _config_write_lock(path):
        data, _persisted = _read_config_pair_unlocked(path)
        return data


def _read_raw_config_dict(path: Path) -> dict[str, Any]:
    """Parse the persisted config without resolving credential references."""

    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8")
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid DeepResearch config JSON: {path}") from error
    if not isinstance(data, dict):
        raise ValueError(f"Invalid DeepResearch config shape: {path}")
    return data


def _read_config_pair_unlocked(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read runtime and persisted forms while the caller holds the config lock."""

    raw = _read_raw_config_dict(path)
    runtime, protected, migrated = resolve_config_secrets(path, raw)
    if not isinstance(runtime, dict) or not isinstance(protected, dict):
        raise ValueError(f"Invalid DeepResearch config shape: {path}")
    if migrated:
        _write_json_config_unlocked(path, protected)
    return runtime, protected


def _write_config_dict(path: Path, data: Mapping[str, Any]) -> None:
    """Atomically replace a config dict under the shared file lock."""

    with _config_write_lock(path):
        _write_config_dict_unlocked(path, data)


def _update_config_dict(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    """Lock the complete read-modify-write cycle for one config file."""

    with _config_write_lock(path):
        data, persisted = _read_config_pair_unlocked(path)
        previous = json.loads(json.dumps(data))
        mutate(data)
        _write_config_dict_unlocked(
            path,
            data,
            previous_plain=previous,
            previous_protected=persisted,
        )


def _write_config_dict_unlocked(
    path: Path,
    data: Mapping[str, Any],
    *,
    previous_plain: Mapping[str, Any] | None = None,
    previous_protected: Mapping[str, Any] | None = None,
) -> None:
    """Protect credentials, then atomically replace the config JSON."""

    protected = protect_config_secrets(
        path,
        dict(data),
        previous_plain=previous_plain,
        previous_protected=previous_protected,
    )
    if not isinstance(protected, dict):  # pragma: no cover - Mapping guarantees this
        raise ValueError(f"Invalid DeepResearch config shape: {path}")
    _write_json_config_unlocked(path, protected)


def _write_json_config_unlocked(path: Path, data: Mapping[str, Any]) -> None:
    """Write already-protected JSON through a sibling temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
    )
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _config_write_lock(path: Path) -> Iterator[None]:
    """Serialize config writers in-process and across API/worker processes."""

    key = str(path.resolve())
    with _CONFIG_LOCKS_GUARD:
        local_lock = _CONFIG_LOCKS.setdefault(key, threading.RLock())
    with local_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_name(f".{path.name}.lock")
        with lock_path.open("a+b") as handle:
            _lock_config_file(handle)
            try:
                yield
            finally:
                _unlock_config_file(handle)


def _lock_config_file(handle) -> None:  # noqa: ANN001
    if os.name == "nt":
        import msvcrt

        handle.seek(0, os.SEEK_END)
        if handle.tell() == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)


def _unlock_config_file(handle) -> None:  # noqa: ANN001
    if os.name == "nt":
        import msvcrt

        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def load_context_budget_config(path: Path):
    """Load the saved context-budget config, or ``None`` when absent/invalid.

    Returns a :class:`ContextBudgetConfig` (DR-ITER-057). Any malformed block is
    treated as unset so resolution falls through to env / built-in defaults.
    """

    from .baseline.context_budget import ContextBudgetConfig, coerce_payload

    try:
        data = _read_config_dict(path)
    except ValueError:
        return None
    block = data.get("context_budget")
    if not isinstance(block, dict):
        return None
    try:
        return coerce_payload(block)
    except ValueError:
        return None


def save_context_budget_config(path: Path, config) -> None:
    """Save the context-budget config, preserving other top-level keys (DR-ITER-057)."""

    _update_config_dict(path, lambda data: data.__setitem__("context_budget", config.to_dict()))


def load_retrieval_settings_config(path: Path):
    """Load retrieval settings, defaulting to disk Chroma persistence."""

    from .baseline.retrieval_settings import coerce_retrieval_settings, default_retrieval_settings

    try:
        data = _read_config_dict(path)
    except ValueError:
        return default_retrieval_settings()
    try:
        return coerce_retrieval_settings(data.get("retrieval"))
    except ValueError:
        return default_retrieval_settings()


def save_retrieval_settings_config(path: Path, config) -> None:
    """Save retrieval settings while preserving other top-level config blocks."""

    _update_config_dict(path, lambda data: data.__setitem__("retrieval", config.to_dict()))


def load_deep_checkpoint_settings_config(path: Path):
    """Load Deep checkpoint settings, defaulting automatic checkpoints to off."""

    from .baseline.deep_checkpoint_settings import (
        coerce_deep_checkpoint_settings,
        default_deep_checkpoint_settings,
    )

    try:
        data = _read_config_dict(path)
    except ValueError:
        return default_deep_checkpoint_settings()
    try:
        return coerce_deep_checkpoint_settings(data.get("deep"))
    except ValueError:
        return default_deep_checkpoint_settings()


def save_deep_checkpoint_settings_config(path: Path, config) -> None:
    """Save Deep checkpoint settings while preserving every other config block."""

    def _mutate(data: dict[str, Any]) -> None:
        existing = data.get("deep")
        deep = dict(existing) if isinstance(existing, dict) else {}
        deep.update(config.to_dict())
        data["deep"] = deep

    _update_config_dict(path, _mutate)


def load_post_run_notes_config(path: Path):
    """Load the saved post-run notes config, defaulting to all modes disabled."""

    from .baseline.post_run_notes_config import coerce_post_run_notes_config, default_post_run_notes_config

    try:
        data = _read_config_dict(path)
    except ValueError:
        return default_post_run_notes_config()
    try:
        return coerce_post_run_notes_config(data.get("post_run_notes"))
    except ValueError:
        return default_post_run_notes_config()


def save_post_run_notes_config(path: Path, config) -> None:
    """Save post-run notes config while preserving other top-level config blocks."""

    _update_config_dict(path, lambda data: data.__setitem__("post_run_notes", config.to_dict()))


def load_memory_settings_config(path: Path):
    """Load saved Memory settings, defaulting to safe built-ins."""

    from .baseline.memory_settings import coerce_memory_settings, default_memory_settings

    try:
        data = _read_config_dict(path)
    except ValueError:
        return default_memory_settings()
    try:
        return coerce_memory_settings(data.get("memory_settings"))
    except ValueError:
        return default_memory_settings()


def save_memory_settings_config(path: Path, config) -> None:
    """Save Memory settings while preserving other top-level config blocks."""

    _update_config_dict(path, lambda data: data.__setitem__("memory_settings", config.to_dict()))


def load_knowledge_graph_config(path: Path):
    """Load Knowledge Graph settings, defaulting to post-run rebuild enabled."""

    from .baseline.knowledge_graph.settings import (
        coerce_knowledge_graph_config,
        default_knowledge_graph_config,
    )

    try:
        data = _read_config_dict(path)
    except ValueError:
        return default_knowledge_graph_config()
    try:
        return coerce_knowledge_graph_config(data.get("knowledge_graph"))
    except ValueError:
        return default_knowledge_graph_config()


def save_knowledge_graph_config(path: Path, config) -> None:
    """Save Knowledge Graph settings while preserving other top-level config blocks."""

    _update_config_dict(path, lambda data: data.__setitem__("knowledge_graph", config.to_dict()))


def normalize_memory_dream_schedule(value: object) -> str:
    """Normalize the saved Dreaming schedule value."""

    schedule = str(value or "off").strip().lower()
    if schedule not in MEMORY_DREAM_SCHEDULES:
        raise ValueError("memory_dream_schedule must be one of: off, daily, weekly")
    return schedule


def load_memory_dream_schedule_config(path: Path) -> str:
    """Load the saved Dreaming schedule, defaulting to off on absent/malformed config."""

    try:
        data = _read_config_dict(path)
    except ValueError:
        return "off"
    try:
        return normalize_memory_dream_schedule(data.get("memory_dream_schedule"))
    except ValueError:
        return "off"


def save_memory_dream_schedule_config(path: Path, schedule: str) -> None:
    """Save the Dreaming schedule while preserving other top-level config blocks."""

    normalized = normalize_memory_dream_schedule(schedule)
    _update_config_dict(path, lambda data: data.__setitem__("memory_dream_schedule", normalized))


def config_has_llm_registry(path: Path) -> bool:
    """Return whether the config file already declares an explicit LLM registry."""

    try:
        data = _read_config_dict(path)
    except ValueError:
        return False
    # Presence of the key (even an empty list) means the registry is explicit:
    # a deliberately-emptied registry must not silently fall back to the flat default.
    return isinstance(data.get("llm_providers"), list)


def registry_from_provider(provider: ProviderSettings) -> LLMRegistry:
    """Synthesize a single-profile registry from flat provider settings.

    Returns an empty registry when the flat LLM triple is not configured.
    """

    if not (_configured_value(provider.llm_base_url) and _configured_value(provider.llm_api_key) and _configured_value(provider.llm_model)):
        return LLMRegistry()
    model = provider.llm_model.strip()
    profile = LLMProviderProfile(
        id="default",
        label="Default",
        type="openai",
        base_url=provider.llm_base_url.strip(),
        api_key=provider.llm_api_key,
        models=(ModelProfile(id=model, kind="chat"),),
        enabled=True,
    )
    return LLMRegistry(providers=(profile,), roles=LLMRoles(default=f"default:{model}"))


def load_llm_registry(path: Path, *, environ: Mapping[str, str] | None = None) -> LLMRegistry:
    """Load the LLM registry, migrating from legacy flat config and merging env.

    Precedence: an explicit ``llm_providers`` block wins; otherwise a single
    ``default`` profile is synthesized from the flat ``provider`` block. Live
    ``DEEPRESEARCH_LLM_*`` env values are merged into the ``default`` profile.
    """

    data = _read_config_dict(path)
    if isinstance(data.get("llm_providers"), list):
        registry = LLMRegistry.from_config(data)
    else:
        provider = data.get("provider", data)
        flat = ProviderSettings(
            **{field: _string_value(provider.get(field)) for field in PROVIDER_FIELDS}
        ) if isinstance(provider, dict) else ProviderSettings()
        registry = registry_from_provider(flat)
    return _merge_env_into_registry(registry, environ)


def save_llm_registry(path: Path, registry: LLMRegistry) -> None:
    """Persist the LLM registry, preserving the ``provider`` (search) block."""

    config = registry.to_config()
    def _mutate(data: dict[str, Any]) -> None:
        data["llm_providers"] = config["llm_providers"]
        data["llm_roles"] = config["llm_roles"]

    _update_config_dict(path, _mutate)


def _merge_env_into_registry(registry: LLMRegistry, environ: Mapping[str, str] | None) -> LLMRegistry:
    """Merge live ``DEEPRESEARCH_LLM_*`` env values into the ``default`` profile."""

    env_provider = provider_settings_from_mapping(environ)
    has_env = any(
        _configured_value(getattr(env_provider, field))
        for field in ("llm_base_url", "llm_api_key", "llm_model")
    )
    if not has_env:
        return registry
    existing = registry.get("default")
    base_url = env_provider.llm_base_url.strip() or (existing.base_url if existing else "")
    api_key = env_provider.llm_api_key.strip() or (existing.api_key if existing else "")
    model = env_provider.llm_model.strip()
    base_models: list[ModelProfile] = list(existing.models) if existing else []
    if model and model not in [m.id for m in base_models]:
        base_models.insert(0, ModelProfile(id=model, kind="chat"))
    profile = LLMProviderProfile(
        id="default",
        label=existing.label if existing else "Default",
        type=existing.type if existing else "openai",
        base_url=base_url,
        api_key=api_key,
        models=tuple(base_models),
        enabled=existing.enabled if existing else True,
    )
    registry = registry.upsert(profile)
    if not registry.roles.default and (model or base_models):
        chosen = model or base_models[0].id
        registry = registry.set_roles(default=f"default:{chosen}")
    return registry


def example_provider_config_text() -> str:
    """Return a placeholder provider config template for first-time local setup."""

    return json.dumps(
        {
            "provider": {
                "llm_base_url": "https://your-llm-provider.example/v1",
                "llm_api_key": "YOUR_LLM_API_KEY",
                "llm_model": "your-model-name",
                "brave_api_key": "YOUR_BRAVE_API_KEY",
                "firecrawl_api_key": "YOUR_FIRECRAWL_API_KEY",
                "exa_api_key": "YOUR_EXA_API_KEY",
                "tavily_api_key": "YOUR_TAVILY_API_KEY",
            }
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n"


def provider_config_template_payload() -> dict[str, Any]:
    """Return a static secret-free setup template payload for local surfaces."""

    return {
        "tool": "deepresearch_config_template",
        "template": example_provider_config_text(),
        "environment_variables": list(PROVIDER_ENV_VARS),
        "config_environment_variable": "DEEPRESEARCH_CONFIG",
        "asset_environment_variables": list(ASSET_ENV_VARS),
        "notes": [
            "模板只包含占位符；真实值应保存在私密环境变量或用户本地配置文件中。",
            "占位符会被诊断视为未配置，替换后可运行本地配置检查。",
            "此模板入口只返回文本，不写入文件，也不读取当前环境中的配置值。",
        ],
        "writes_files": False,
    }


def setup_provider_config(path: Path, *, source: str) -> dict[str, Any]:
    """Create a placeholder config when missing and return a secret-free setup payload."""

    created = False
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("x", encoding="utf-8") as file:
                file.write(example_provider_config_text())
            created = True
        except FileExistsError:
            created = False
    provider = load_provider_config(path)
    return provider_config_setup_payload(path, source=source, provider=provider, created=created)


def provider_config_setup_payload(path: Path, *, source: str, provider: ProviderSettings, created: bool) -> dict[str, Any]:
    """Return a secret-free config setup summary for first-run surfaces."""

    return {
        "tool": "deepresearch_config_setup",
        "config_path": str(path),
        "config_source": source,
        "exists": path.exists(),
        "created": bool(created),
        "provider_status": provider.safe_status(),
        "missing_required": provider.missing_required(),
        "next_steps": [
            "Edit this local config file on your machine and replace placeholders with private BYOK values.",
            "Run deepresearch doctor to verify local setup without a live provider check.",
            "Run doctor --connectivity-check only when you explicitly want to test the private LLM provider live.",
        ],
        "writes_files": bool(created),
    }


def format_config_setup_text(payload: Mapping[str, Any]) -> str:
    """Return a human-readable config setup summary without secret values."""

    lines = [
        "DeepResearch local setup",
        f"Config file: {payload.get('config_path', '')}",
        f"Source: {payload.get('config_source', '')}",
        f"Created: {bool(payload.get('created'))}",
        f"Exists: {bool(payload.get('exists'))}",
        f"Provider: {payload.get('provider_status', {})}",
    ]
    missing = payload.get("missing_required") if isinstance(payload.get("missing_required"), list) else []
    if missing:
        lines.append(f"Missing: {', '.join(str(item) for item in missing)}")
    steps = payload.get("next_steps") if isinstance(payload.get("next_steps"), list) else []
    if steps:
        lines.append("Next steps:")
        lines.extend(f"- {step}" for step in steps)
    return "\n".join(lines)


def merge_provider_config(base: ProviderSettings, updates: Mapping[str, str]) -> ProviderSettings:
    """Return provider settings with non-empty update values applied."""

    values: dict[str, str] = {}
    for field in PROVIDER_FIELDS:
        update = _string_value(updates.get(field))
        values[field] = update or str(getattr(base, field) or "")
    return ProviderSettings(**values)


def provider_config_updates(values: Mapping[str, Any]) -> dict[str, str]:
    """Return non-empty provider config values from parsed CLI fields."""

    updates: dict[str, str] = {}
    for field in PROVIDER_FIELDS:
        value = _string_value(values.get(field))
        if value:
            updates[field] = value
    return updates


def _string_value(value: Any) -> str:
    """Return a stripped string value for config fields."""

    return str(value or "").strip()
