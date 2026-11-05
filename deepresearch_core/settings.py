import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


_PLACEHOLDER_VALUES = {
    "your_llm_api_key",
    "your_brave_api_key",
    "your_firecrawl_api_key",
    "your_exa_api_key",
    "your_tavily_api_key",
    "your-model-name",
    "model-name",
    "your-model",
}
_PLACEHOLDER_FRAGMENTS = (
    "your-llm-provider.example",
    "api.example.com",
)
PROVIDER_SEMANTIC_VERIFICATION_ENV = "DEEPRESEARCH_PROVIDER_SEMANTIC_VERIFICATION"
_TRUTHY_VALUES = {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class ResearchSettings:
    """Local runtime settings shared by all DeepResearch surfaces."""

    reports_dir: Path
    provider_semantic_verification: bool = False


@dataclass(frozen=True)
class ProviderSettings:
    """BYOK provider settings shared by API, CLI, MCP, and desktop surfaces."""

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    brave_api_key: str = ""
    firecrawl_api_key: str = ""
    firecrawl_base_url: str = "https://api.firecrawl.dev"
    exa_api_key: str = ""
    tavily_api_key: str = ""

    def missing_required(self) -> list[str]:
        """Return user-facing names for missing required settings."""

        missing: list[str] = []
        if not _configured_value(self.llm_base_url):
            missing.append("LLM Base URL")
        if not _configured_value(self.llm_api_key):
            missing.append("LLM API Key")
        if not _configured_value(self.llm_model):
            missing.append("LLM Model")
        if not any(
            _configured_value(value)
            for value in (self.brave_api_key, self.tavily_api_key, self.exa_api_key, self.firecrawl_api_key)
        ):
            missing.append("Search Provider API Key")
        return missing

    def safe_summary(self) -> str:
        """Return a secret-free provider status summary."""

        return repr(self.safe_status())

    def safe_status(self) -> dict[str, bool]:
        """Return secret-free field readiness booleans for provider settings."""

        return {
            "llm_base_url": _configured_value(self.llm_base_url),
            "llm_api_key": _configured_value(self.llm_api_key),
            "llm_model": _configured_value(self.llm_model),
            "brave_api_key": _configured_value(self.brave_api_key),
            "firecrawl_api_key": _configured_value(self.firecrawl_api_key),
            "firecrawl_base_url": _configured_value(self.firecrawl_base_url),
            "exa_api_key": _configured_value(self.exa_api_key),
            "tavily_api_key": _configured_value(self.tavily_api_key),
        }


def _configured_value(value: str) -> bool:
    """Return whether a config value looks user-provided instead of a template placeholder."""

    text = str(value or "").strip()
    if not text:
        return False
    normalized = text.lower().replace("-", "_")
    if normalized in _PLACEHOLDER_VALUES:
        return False
    if normalized.startswith("your_") or normalized.startswith("replace_"):
        return False
    lowered = text.lower()
    return not any(fragment in lowered for fragment in _PLACEHOLDER_FRAGMENTS)


def provider_semantic_verification_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Return whether provider-backed semantic verification is explicitly enabled."""

    env = environ if environ is not None else os.environ
    return str(env.get(PROVIDER_SEMANTIC_VERIFICATION_ENV) or "").strip().lower() in _TRUTHY_VALUES
