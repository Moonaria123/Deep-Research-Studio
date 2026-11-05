import os
from collections.abc import Mapping

from .settings import ProviderSettings


ENV_ALIASES = {
    "llm_base_url": ("DEEPRESEARCH_LLM_BASE_URL", "OPENAI_BASE_URL"),
    "llm_api_key": ("DEEPRESEARCH_LLM_API_KEY", "OPENAI_API_KEY"),
    "llm_model": ("DEEPRESEARCH_LLM_MODEL", "OPENAI_MODEL"),
    "brave_api_key": ("DEEPRESEARCH_BRAVE_API_KEY", "BRAVE_API_KEY"),
    "firecrawl_api_key": ("DEEPRESEARCH_FIRECRAWL_API_KEY", "FIRECRAWL_API_KEY"),
    "firecrawl_base_url": ("DEEPRESEARCH_FIRECRAWL_BASE_URL", "FIRECRAWL_BASE_URL"),
    "exa_api_key": ("DEEPRESEARCH_EXA_API_KEY", "EXA_API_KEY"),
    "tavily_api_key": ("DEEPRESEARCH_TAVILY_API_KEY", "TAVILY_API_KEY"),
}


def provider_settings_from_mapping(
    environ: Mapping[str, str] | None = None,
    *,
    overrides: Mapping[str, str] | None = None,
    defaults: ProviderSettings | Mapping[str, str] | None = None,
) -> ProviderSettings:
    """Build provider settings from explicit overrides and environment values."""

    env = environ if environ is not None else os.environ
    override_values = overrides or {}
    default_values = defaults or {}
    return ProviderSettings(
        llm_base_url=_value("llm_base_url", env, override_values, default_values),
        llm_api_key=_value("llm_api_key", env, override_values, default_values),
        llm_model=_value("llm_model", env, override_values, default_values),
        brave_api_key=_value("brave_api_key", env, override_values, default_values),
        firecrawl_api_key=_value("firecrawl_api_key", env, override_values, default_values),
        firecrawl_base_url=_value("firecrawl_base_url", env, override_values, default_values) or "https://api.firecrawl.dev",
        exa_api_key=_value("exa_api_key", env, override_values, default_values),
        tavily_api_key=_value("tavily_api_key", env, override_values, default_values),
    )


def _value(
    field: str,
    environ: Mapping[str, str],
    overrides: Mapping[str, str],
    defaults: ProviderSettings | Mapping[str, str],
) -> str:
    """Return an explicit override before falling back to env and defaults."""

    override = overrides.get(field, "").strip()
    if override:
        return override
    for name in ENV_ALIASES[field]:
        value = environ.get(name, "").strip()
        if value:
            return value
    if isinstance(defaults, ProviderSettings):
        return str(getattr(defaults, field) or "").strip()
    value = str(defaults.get(field, "") or "").strip()
    if value:
        return value
    return ""
