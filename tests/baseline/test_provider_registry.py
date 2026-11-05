"""LLM provider registry: data model, CRUD, role bindings, secret-free status, migration."""

from __future__ import annotations

import json

import pytest

from deepresearch_core.config import (
    config_has_llm_registry,
    load_llm_registry,
    registry_from_provider,
    save_llm_registry,
)
from deepresearch_core.providers_registry import (
    LLMProviderProfile,
    LLMRegistry,
    LLMRoles,
    ModelCapabilities,
    ModelProfile,
)
from deepresearch_core.settings import ProviderSettings


# --- ModelProfile (rich model schema) -----------------------------------------


def test_model_profile_defaults_to_chat_text():
    m = ModelProfile(id="gpt-4o")
    assert m.kind == "chat"
    assert m.is_embed is False
    assert m.is_multimodal is False
    assert m.context_window == 128000
    assert m.enabled is True
    assert m.display_name() == "gpt-4o"


def test_model_profile_alias_and_multimodal():
    m = ModelProfile(
        id="gpt-4o",
        alias="GPT-4o",
        capabilities=ModelCapabilities(vision=True, web_search=True, thinking=True),
        thinking_budget=2048,
    )
    assert m.display_name() == "GPT-4o"
    assert m.is_multimodal is True
    cfg = m.to_config()
    assert cfg["alias"] == "GPT-4o"
    assert cfg["capabilities"] == {"vision": True, "ocr": False, "web_search": True, "thinking": True}
    assert cfg["thinking_budget"] == 2048


def test_model_profile_embed_carries_dimensions():
    m = ModelProfile(id="text-embed-3", kind="embed", dimensions=1536)
    assert m.is_embed is True
    cfg = m.to_config()
    assert cfg["kind"] == "embed"
    assert cfg["dimensions"] == 1536


def test_model_profile_from_legacy_string():
    m = ModelProfile.from_config("gpt-4o")
    assert m.id == "gpt-4o" and m.kind == "chat"


def test_model_profile_from_config_clamps_unknown_kind():
    m = ModelProfile.from_config({"id": "x", "kind": "bogus"})
    assert m.kind == "chat"


def test_model_profile_roundtrip_through_config():
    original = ModelProfile(
        id="m1", alias="M1", kind="chat", context_window=8000,
        capabilities=ModelCapabilities(ocr=True), thinking_budget=512,
    )
    restored = ModelProfile.from_config(original.to_config())
    assert restored == original


def test_model_profile_temperature_roundtrip_and_clamp():
    original = ModelProfile(id="m1", temperature=0.7)
    cfg = original.to_config()
    assert cfg["temperature"] == 0.7
    assert ModelProfile.from_config(cfg).temperature == 0.7

    # Absent temperature is omitted from config and parses back to None.
    bare = ModelProfile(id="m2")
    assert "temperature" not in bare.to_config()
    assert ModelProfile.from_config({"id": "m2"}).temperature is None

    # Out-of-range values clamp to [0, 2]; junk parses to None.
    assert ModelProfile.from_config({"id": "m3", "temperature": 5}).temperature == 2.0
    assert ModelProfile.from_config({"id": "m4", "temperature": "abc"}).temperature is None


def test_provider_models_coerce_legacy_strings():
    profile = LLMProviderProfile(id="p", api_key="k", models=("a", "b", ""))
    assert profile.model_ids() == ("a", "b")  # empties dropped
    assert all(isinstance(m, ModelProfile) for m in profile.models)


def test_provider_models_accept_rich_dicts():
    profile = LLMProviderProfile.from_config({
        "id": "p", "api_key": "k",
        "models": [{"id": "e", "kind": "embed", "dimensions": 768}, "c"],
    })
    embed = profile.get_model("e")
    assert embed is not None and embed.is_embed and embed.dimensions == 768
    assert profile.get_model("c").kind == "chat"


def test_provider_status_models_have_no_secrets_and_full_shape():
    profile = LLMProviderProfile(
        id="p", api_key="sk-secret",
        models=({"id": "m", "alias": "M", "capabilities": {"vision": True}},),
    )
    status = profile.status()
    assert "sk-secret" not in json.dumps(status)
    m0 = status["models"][0]
    assert m0["id"] == "m" and m0["alias"] == "M"
    assert m0["capabilities"]["vision"] is True


# --- data model / CRUD --------------------------------------------------------


def test_upsert_adds_and_updates_profile():
    reg = LLMRegistry()
    reg = reg.upsert(
        LLMProviderProfile(id="openai", label="OpenAI", type="openai", base_url="https://api.openai.com/v1", api_key="sk-1", models=("gpt-4o",))
    )
    assert reg.get("openai").label == "OpenAI"
    assert len(reg.providers) == 1

    reg = reg.upsert(
        LLMProviderProfile(id="openai", label="OpenAI 2", type="openai", base_url="https://api.openai.com/v1", api_key="sk-2", models=("gpt-4o", "gpt-4o-mini"))
    )
    assert len(reg.providers) == 1
    assert reg.get("openai").label == "OpenAI 2"
    assert reg.get("openai").model_ids() == ("gpt-4o", "gpt-4o-mini")


def test_upsert_requires_id():
    with pytest.raises(ValueError):
        LLMRegistry().upsert(LLMProviderProfile(id="", base_url="https://x/v1"))


def test_remove_profile_scrubs_role_bindings():
    reg = LLMRegistry(
        providers=(
            LLMProviderProfile(id="a", api_key="k", models=("m1",)),
            LLMProviderProfile(id="b", api_key="k", models=("m2",)),
        ),
        roles=LLMRoles(default="a:m1", embed="a:m1", council=("a:m1", "b:m2")),
    )
    reg = reg.set_roles(memory="a:m1")
    reg = reg.remove("a")
    assert reg.get("a") is None
    assert reg.roles.default == ""
    assert reg.roles.embed == ""
    assert reg.roles.memory == ""
    assert reg.roles.council == ("b:m2",)


def test_remove_model_scrubs_memory_role():
    reg = LLMRegistry(
        providers=(LLMProviderProfile(id="a", api_key="k", models=("m1", "m2")),),
        roles=LLMRoles(default="a:m1", embed="a:m2", memory="a:m1", council=("a:m1", "a:m2")),
    )

    reg = reg.remove_model("a", "m1")

    assert reg.roles.default == ""
    assert reg.roles.embed == "a:m2"
    assert reg.roles.memory == ""
    assert reg.roles.council == ("a:m2",)


def test_set_roles_validates_provider_exists():
    reg = LLMRegistry(providers=(LLMProviderProfile(id="a", api_key="k", models=("m1",)),))
    with pytest.raises(ValueError):
        reg.set_roles(default="ghost:x")
    reg2 = reg.set_roles(default="a:m1", memory="a:m1", council=("a:m1",))
    assert reg2.roles.default == "a:m1"
    assert reg2.roles.memory == "a:m1"
    assert reg2.roles.council == ("a:m1",)


def test_resolve_ref_splits_on_first_colon():
    reg = LLMRegistry(
        providers=(LLMProviderProfile(id="ol", type="ollama", base_url="http://127.0.0.1:11434", models=("llama3.1:8b",)),)
    )
    resolved = reg.resolve("ol:llama3.1:8b")
    assert resolved is not None
    profile, model = resolved
    assert profile.id == "ol"
    assert model == "llama3.1:8b"
    assert reg.resolve("ghost:x") is None
    assert reg.resolve("") is None


# --- secret-free status -------------------------------------------------------


def test_status_is_secret_free():
    reg = LLMRegistry(
        providers=(
            LLMProviderProfile(id="a", label="A", type="openai", base_url="https://x/v1", api_key="sk-secret", models=("m1",), enabled=True),
        ),
        roles=LLMRoles(default="a:m1"),
    )
    status = reg.status()
    blob = json.dumps(status)
    assert "sk-secret" not in blob
    prov = status["providers"][0]
    assert prov["has_key"] is True
    assert prov["id"] == "a"
    assert prov["base_url"] == "https://x/v1"
    assert [m["id"] for m in prov["models"]] == ["m1"]
    assert "api_key" not in prov
    assert status["roles"]["default"] == "a:m1"


def test_status_placeholder_key_not_configured():
    reg = LLMRegistry(providers=(LLMProviderProfile(id="a", api_key="your_llm_api_key", models=()),))
    assert reg.status()["providers"][0]["has_key"] is False


# --- migration + persistence (config.py) --------------------------------------


def test_load_llm_registry_migrates_flat_config(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "provider": {
                    "llm_base_url": "https://api.openai.com/v1",
                    "llm_api_key": "sk-flat",
                    "llm_model": "gpt-4o",
                    "brave_api_key": "b",
                }
            }
        ),
        encoding="utf-8",
    )
    reg = load_llm_registry(path)
    assert len(reg.providers) == 1
    profile = reg.providers[0]
    assert profile.id == "default"
    assert profile.base_url == "https://api.openai.com/v1"
    assert profile.api_key == "sk-flat"
    assert "gpt-4o" in profile.model_ids()
    assert reg.roles.default == "default:gpt-4o"


def test_load_llm_registry_reads_registry_shape(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "llm_providers": [
                    {
                        "id": "openai",
                        "label": "OpenAI",
                        "type": "openai",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "sk-x",
                        "models": ["gpt-4o"],
                        "enabled": True,
                    }
                ],
                "llm_roles": {"default": "openai:gpt-4o", "embed": "", "memory": "openai:gpt-4o", "council": ["openai:gpt-4o"]},
            }
        ),
        encoding="utf-8",
    )
    assert config_has_llm_registry(path) is True
    reg = load_llm_registry(path)
    assert reg.get("openai").api_key == "sk-x"
    assert reg.roles.default == "openai:gpt-4o"
    assert reg.roles.memory == "openai:gpt-4o"
    assert reg.roles.council == ("openai:gpt-4o",)


def test_save_then_load_round_trip_preserves_provider_block(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"provider": {"brave_api_key": "brave-1", "firecrawl_api_key": "fc-1"}}),
        encoding="utf-8",
    )
    reg = LLMRegistry(
        providers=(LLMProviderProfile(id="openai", type="openai", base_url="https://api.openai.com/v1", api_key="sk-x", models=("gpt-4o",)),),
        roles=LLMRoles(default="openai:gpt-4o"),
    )
    save_llm_registry(path, reg)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data["provider"]["brave_api_key"]) == {"secret_ref"}  # encrypted search ref preserved
    assert "brave-1" not in path.read_text(encoding="utf-8")
    assert data["llm_providers"][0]["id"] == "openai"

    reloaded = load_llm_registry(path)
    assert reloaded.get("openai").api_key == "sk-x"
    assert reloaded.roles.default == "openai:gpt-4o"


def test_save_provider_config_preserves_llm_registry(tmp_path):
    from deepresearch_core.config import save_provider_config

    path = tmp_path / "config.json"
    reg = LLMRegistry(
        providers=(LLMProviderProfile(id="openai", type="openai", base_url="https://x/v1", api_key="sk-x", models=("gpt-4o",)),),
        roles=LLMRoles(default="openai:gpt-4o"),
    )
    save_llm_registry(path, reg)
    save_provider_config(path, ProviderSettings(brave_api_key="brave-z"))

    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data["provider"]["brave_api_key"]) == {"secret_ref"}
    assert "brave-z" not in path.read_text(encoding="utf-8")
    assert data["llm_providers"][0]["id"] == "openai"  # registry survives a flat save


def test_env_merge_builds_default_profile(tmp_path):
    path = tmp_path / "config.json"  # absent
    env = {
        "DEEPRESEARCH_LLM_BASE_URL": "https://env/v1",
        "DEEPRESEARCH_LLM_API_KEY": "sk-env",
        "DEEPRESEARCH_LLM_MODEL": "env-model",
    }
    reg = load_llm_registry(path, environ=env)
    assert reg.get("default").base_url == "https://env/v1"
    assert reg.get("default").api_key == "sk-env"
    assert reg.roles.default == "default:env-model"


def test_config_has_llm_registry_false_for_flat(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"provider": {"llm_api_key": "sk"}}), encoding="utf-8")
    assert config_has_llm_registry(path) is False


def test_emptied_registry_is_explicit_and_not_resurrected(tmp_path):
    """Removing the last provider must persist an explicit empty registry.

    Even when a legacy flat ``provider`` block is still present, an explicitly
    saved empty ``llm_providers`` list must NOT trigger the flat-default fallback.
    """

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "provider": {
                    "llm_base_url": "https://api.openai.com/v1",
                    "llm_api_key": "sk-flat",
                    "llm_model": "gpt-4o",
                }
            }
        ),
        encoding="utf-8",
    )
    save_llm_registry(path, LLMRegistry())  # user cleared all providers

    assert config_has_llm_registry(path) is True
    reloaded = load_llm_registry(path)
    assert reloaded.providers == ()


def test_registry_from_provider_synthesizes_default():
    provider = ProviderSettings(llm_base_url="https://x/v1", llm_api_key="sk", llm_model="m1")
    reg = registry_from_provider(provider)
    assert reg.get("default").base_url == "https://x/v1"
    assert reg.roles.default == "default:m1"
    only = reg.get("default").models[0]
    assert only.id == "m1" and only.kind == "chat"

    empty = registry_from_provider(ProviderSettings())
    assert empty.providers == ()
    assert empty.roles.default == ""


def test_env_merge_does_not_duplicate_existing_model(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "llm_providers": [
                    {
                        "id": "default", "type": "openai", "base_url": "https://x/v1",
                        "api_key": "sk-x", "models": [{"id": "m1", "kind": "chat"}],
                    }
                ],
                "llm_roles": {"default": "default:m1"},
            }
        ),
        encoding="utf-8",
    )
    env = {"DEEPRESEARCH_LLM_MODEL": "m1"}  # same model arrives via env
    reg = load_llm_registry(path, environ=env)
    assert reg.get("default").model_ids() == ("m1",)  # not duplicated
