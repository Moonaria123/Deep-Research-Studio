"""Run the DeepResearch baseline API gateway (FastAPI + uvicorn).

BYOK precedence: environment variables override the user-local config file. No secret
is ever logged or returned by the API (only secret-free provider status).

Usage:
    pip install fastapi uvicorn
    python run_api.py            # serves http://127.0.0.1:8787/api/v1
Env (optional, override config file):
    DEEPRESEARCH_LLM_BASE_URL, DEEPRESEARCH_LLM_API_KEY, DEEPRESEARCH_LLM_MODEL,
    DEEPRESEARCH_BRAVE_API_KEY, DEEPRESEARCH_FIRECRAWL_API_KEY, DEEPRESEARCH_FIRECRAWL_BASE_URL
    DEEPRESEARCH_DB_PATH (default: <APPDATA>/DeepResearch/baseline.sqlite3)
    DEEPRESEARCH_API_HOST (default 127.0.0.1), DEEPRESEARCH_API_PORT (default 8787)
    DEEPRESEARCH_PREFER_LOCAL_MODELS=1 to try local Ollama first
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

from deepresearch_core.baseline import BaselineService, create_app
from deepresearch_core.config import (
    PROVIDER_ENV_VARS,
    PROVIDER_FIELDS,
    default_config_path,
    load_provider_config,
)
from deepresearch_core.observability import configure_logging, get_logger
from deepresearch_core.settings import ProviderSettings
from deepresearch_core.secret_vault import SecretVaultError, validate_vault_key




def _load_settings() -> ProviderSettings:
    """Load provider settings with precedence: env > DPAPI secure config > plaintext.

    The Tauri desktop shell injects secrets via env (sourced from its Stronghold vault),
    which therefore wins. On Windows, an optional DPAPI-encrypted secure config is layered
    over the plaintext config file. No secret is logged.
    """

    return _load_settings_with_sources()[0]


def _load_settings_with_sources() -> tuple[ProviderSettings, dict[str, str]]:
    """Load effective settings plus each search credential's owning layer."""

    key_fields = {
        "brave": "brave_api_key",
        "tavily": "tavily_api_key",
        "exa": "exa_api_key",
        "firecrawl": "firecrawl_api_key",
    }
    try:
        base = load_provider_config(default_config_path())
    except SecretVaultError:
        raise
    except Exception:  # noqa: BLE001 - normalize without exposing private diagnostics
        raise SecretVaultError("Provider configuration could not be loaded safely") from None
    sources = {
        provider: ("config" if str(getattr(base, field, "") or "").strip() else "none")
        for provider, field in key_fields.items()
    }

    # Optional DPAPI secure config overlay (Windows), if present.
    try:
        from deepresearch_core.secure_config import (
            default_secure_config_path,
            load_secure_provider_config,
            secure_config_supported,
        )

        if secure_config_supported():
            secure_path = default_secure_config_path()
            if secure_path.exists():
                secure = load_secure_provider_config(secure_path)
                secure_overlay = {
                    field: value
                    for field in PROVIDER_FIELDS
                    if (value := str(getattr(secure, field, "") or "").strip())
                }
                if secure_overlay:
                    base = dataclasses.replace(base, **secure_overlay)
                    for provider, field in key_fields.items():
                        if field in secure_overlay:
                            sources[provider] = "secure"
    except Exception:  # noqa: BLE001 - secure config is best-effort
        pass

    overrides = {
        field: value
        for field, env in zip(PROVIDER_FIELDS, PROVIDER_ENV_VARS)
        if (value := str(os.environ.get(env) or "").strip())
    }
    if overrides:
        base = dataclasses.replace(base, **overrides)
        for provider, field in key_fields.items():
            if field in overrides:
                sources[provider] = "env"
    return base, sources


def _db_path() -> str:
    explicit = str(os.environ.get("DEEPRESEARCH_DB_PATH") or "").strip()
    if explicit:
        return explicit
    appdata = str(os.environ.get("APPDATA") or "").strip()
    base_dir = Path(appdata) / "DeepResearch" if appdata else Path.home() / ".deepresearch"
    base_dir.mkdir(parents=True, exist_ok=True)
    return str(base_dir / "baseline.sqlite3")


def build_service() -> BaselineService:
    """Construct the BaselineService from environment + config."""

    if os.environ.get("DEEPRESEARCH_VAULT_KEY_FILE") or os.environ.get("DEEPRESEARCH_LOCAL_WEB_GUARD") == "1":
        validate_vault_key(default_config_path())
    provider, credential_sources = _load_settings_with_sources()
    return BaselineService(
        provider=provider,
        db_path=_db_path(),
        prefer_local_models=str(os.environ.get("DEEPRESEARCH_PREFER_LOCAL_MODELS") or "").strip() in {"1", "true", "yes"},
        db_encryption_key=str(os.environ.get("DEEPRESEARCH_DB_KEY") or "").strip() or None,
        config_path=default_config_path(),
        credential_sources=credential_sources,
    )


def main() -> None:
    """Start the uvicorn server."""

    import uvicorn

    configure_logging()
    log = get_logger("deepresearch.api")
    service = build_service()
    app = create_app(service)
    host = str(os.environ.get("DEEPRESEARCH_API_HOST") or "127.0.0.1")
    port = int(os.environ.get("DEEPRESEARCH_API_PORT") or "8787")
    status = service.provider_status()
    log.info(
        "api_start",
        url=f"http://{host}:{port}/api/v1",
        llm_ready=bool(status["has_llm_key"]),
        brave_ready=bool(status["brave_ready"]),
        firecrawl_ready=bool(status["firecrawl_ready"]),
        db_encrypted=service.store.encrypted,
    )
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
