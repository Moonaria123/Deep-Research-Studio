"""Phase 2: registry-driven resolver/router + per-run model/council overrides.

These tests are hermetic: no live network. Service construction only builds
adapter objects (no calls), and orchestrator runs use ``FakeLLM`` members.
"""

from __future__ import annotations

import json
from pathlib import Path

from deepresearch_core.baseline.budget import Mode, RunRequest
from deepresearch_core.baseline.indexing import Indexer
from deepresearch_core.baseline.memory import MemoryEngine
from deepresearch_core.baseline.model_router import ModelRouter, OllamaLLMAdapter
from deepresearch_core.baseline.orchestrator import Orchestrator
from deepresearch_core.baseline.retrieval import Retriever
from deepresearch_core.baseline.service import BaselineService
from deepresearch_core.baseline.store import Store
from deepresearch_core.settings import ProviderSettings

from tests.baseline.fakes import FakeLLM


def _model_of(adapter) -> str | None:
    """Return the model name from an OpenAI-compat or Ollama adapter."""

    if hasattr(adapter, "_client"):
        return getattr(adapter._client, "model", None)
    return getattr(adapter, "model", None)


def _write_registry(path: Path, *, providers: list[dict], roles: dict) -> None:
    path.write_text(
        json.dumps({"llm_providers": providers, "llm_roles": roles}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _service(tmp_path: Path, *, providers: list[dict], roles: dict) -> BaselineService:
    cfg = tmp_path / "config.json"
    _write_registry(cfg, providers=providers, roles=roles)
    return BaselineService(provider=ProviderSettings(), db_path=":memory:", config_path=cfg)


# --- service-level registry resolution -----------------------------------------


def test_default_role_selects_registry_model(tmp_path):
    svc = _service(
        tmp_path,
        providers=[
            {"id": "p1", "type": "openai", "base_url": "https://p1.example/v1",
             "api_key": "sk-p1", "models": ["gpt-4o", "gpt-4o-mini"], "enabled": True},
        ],
        roles={"default": "p1:gpt-4o-mini", "embed": "", "council": []},
    )
    eff = svc._effective_default_provider()
    assert eff.llm_model == "gpt-4o-mini"
    assert eff.llm_base_url == "https://p1.example/v1"
    assert eff.llm_api_key == "sk-p1"


def test_council_members_built_from_roles(tmp_path):
    svc = _service(
        tmp_path,
        providers=[
            {"id": "p1", "type": "openai", "base_url": "https://p1.example/v1",
             "api_key": "sk-p1", "models": ["gpt-4o"], "enabled": True},
            {"id": "p2", "type": "openai", "base_url": "https://p2.example/v1",
             "api_key": "sk-p2", "models": ["claude-x"], "enabled": True},
        ],
        roles={"default": "p1:gpt-4o", "embed": "", "council": ["p1:gpt-4o", "p2:claude-x"]},
    )
    members = svc._orchestrator._council_members
    assert len(members) == 2
    assert {_model_of(m) for m in members} == {"gpt-4o", "claude-x"}


def test_council_single_model_expands_to_multiple_perspectives(tmp_path):
    """A single candidate model fills ``council_size`` members with distinct lenses."""

    svc = _service(
        tmp_path,
        providers=[
            {"id": "p1", "type": "openai", "base_url": "https://p1.example/v1",
             "api_key": "sk-p1", "models": ["gpt-4o"], "enabled": True},
        ],
        roles={"default": "p1:gpt-4o", "embed": "", "council": ["p1:gpt-4o", "p1:gpt-4o"]},
    )
    members = svc._resolve_run_council(
        RunRequest(query="q", council_models=["p1:gpt-4o", "p1:gpt-4o"], council_size=3)
    )
    # One distinct adapter, replicated across 3 members, each with its own perspective.
    assert len(members) == 3
    assert len({_model_of(m.adapter) for m in members}) == 1
    assert len({m.perspective for m in members}) == 3


def test_disabled_profile_excluded_from_council(tmp_path):
    svc = _service(
        tmp_path,
        providers=[
            {"id": "p1", "type": "openai", "base_url": "https://p1.example/v1",
             "api_key": "sk-p1", "models": ["gpt-4o"], "enabled": True},
            {"id": "p2", "type": "openai", "base_url": "https://p2.example/v1",
             "api_key": "sk-p2", "models": ["claude-x"], "enabled": False},
        ],
        roles={"default": "p1:gpt-4o", "embed": "", "council": ["p1:gpt-4o", "p2:claude-x"]},
    )
    members = svc._orchestrator._council_members
    assert len(members) == 1
    assert _model_of(members[0]) == "gpt-4o"


def test_adapter_for_ref_resolves_openai_ollama_and_disabled(tmp_path):
    svc = _service(
        tmp_path,
        providers=[
            {"id": "p1", "type": "openai", "base_url": "https://p1.example/v1",
             "api_key": "sk-p1", "models": ["gpt-4o"], "enabled": True},
            {"id": "local", "type": "ollama", "base_url": "http://127.0.0.1:11434",
             "api_key": "", "models": ["llama3.1:8b"], "enabled": True},
            {"id": "off", "type": "openai", "base_url": "https://off.example/v1",
             "api_key": "sk-off", "models": ["m"], "enabled": False},
        ],
        roles={"default": "p1:gpt-4o", "embed": "", "council": []},
    )
    openai_adapter = svc._adapter_for_ref("p1:gpt-4o")
    assert openai_adapter is not None and _model_of(openai_adapter) == "gpt-4o"

    ollama_adapter = svc._adapter_for_ref("local:llama3.1:8b")
    assert isinstance(ollama_adapter, OllamaLLMAdapter)
    assert ollama_adapter.model == "llama3.1:8b"

    assert svc._adapter_for_ref("off:m") is None
    assert svc._adapter_for_ref("nope:m") is None


def test_ollama_availability_requires_the_selected_model(monkeypatch):
    adapter = OllamaLLMAdapter(model="llama3.1")
    monkeypatch.setattr(adapter, "_get", lambda path: {"models": []})

    assert adapter.available() is False


def test_ollama_availability_accepts_latest_tag_alias(monkeypatch):
    adapter = OllamaLLMAdapter(model="llama3.1")
    monkeypatch.setattr(
        adapter,
        "_get",
        lambda path: {"models": [{"name": "llama3.1:latest"}]},
    )

    assert adapter.available() is True


def test_per_run_resolvers_return_overrides(tmp_path):
    svc = _service(
        tmp_path,
        providers=[
            {"id": "p1", "type": "openai", "base_url": "https://p1.example/v1",
             "api_key": "sk-p1", "models": ["gpt-4o"], "enabled": True},
            {"id": "p2", "type": "openai", "base_url": "https://p2.example/v1",
             "api_key": "sk-p2", "models": ["claude-x"], "enabled": True},
        ],
        roles={"default": "p1:gpt-4o", "embed": "", "council": []},
    )
    llm = svc._resolve_run_llm(RunRequest(query="q", llm_model="p2:claude-x"))
    assert llm is not None and _model_of(llm) == "claude-x"
    assert isinstance(llm, ModelRouter)
    assert llm.owns_fallback is True
    assert len(llm.backends) == 1
    assert getattr(llm, "_deepresearch_model_ref", "") == "p2:claude-x"
    assert svc._resolve_run_llm(RunRequest(query="q")) is None

    council = svc._resolve_run_council(
        RunRequest(query="q", council_models=["p1:gpt-4o", "p2:claude-x"], council_size=2)
    )
    assert len(council) == 2
    assert {_model_of(m.adapter) for m in council} == {"gpt-4o", "claude-x"}
    # No candidate pool => the chair falls back to the default-role model (graceful),
    # still producing council_size members from it.
    fallback = svc._resolve_run_council(RunRequest(query="q", council_size=3))
    assert len(fallback) == 3
    assert {_model_of(m.adapter) for m in fallback} == {"gpt-4o"}


# --- orchestrator per-run fan-out (end-to-end with fakes) ----------------------


def _orchestrator(fake_llm, fake_search, *, llm_resolver=None, council_resolver=None):
    store = Store(":memory:")
    return (
        Orchestrator(
            store=store,
            llm=fake_llm,
            retriever=Retriever(store),
            indexer=Indexer(store),
            search=fake_search,
            memory=MemoryEngine(store),
            enforce_grounding_gate=True,
            llm_resolver=llm_resolver,
            council_resolver=council_resolver,
        ),
        store,
    )


def test_council_resolver_overrides_member_fanout(fake_llm, fake_search):
    members = [FakeLLM("run-a"), FakeLLM("run-b")]
    orch, store = _orchestrator(
        fake_llm, fake_search, council_resolver=lambda req: members
    )
    sid = store.create_session(title="t")
    outcome = orch.run(
        RunRequest(query="quantum computing qubits", mode=Mode.COUNCIL, session_id=sid)
    )
    payload = outcome.to_dict()
    assert payload["status"] == "completed"
    assert len(payload["members"]) == 2
    store.close()


def test_llm_resolver_overrides_default_composer(fake_llm, fake_search):
    run_llm = FakeLLM("per-run")
    orch, store = _orchestrator(
        fake_llm, fake_search, llm_resolver=lambda req: run_llm
    )
    sid = store.create_session(title="t")
    outcome = orch.run(
        RunRequest(query="quantum computing qubits", mode=Mode.FAST, session_id=sid)
    )
    assert outcome.to_dict()["status"] == "completed"
    # The per-run LLM did the composing; the base llm was not invoked.
    assert run_llm.calls
    store.close()
