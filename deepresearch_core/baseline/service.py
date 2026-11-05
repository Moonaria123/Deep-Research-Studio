"""BaselineService: wires the baseline subsystem from BYOK provider settings.

This is the framework-agnostic integration point used by the API, CLI, MCP, and tests.
Runs execute in background threads; events are persisted to the store so streaming and
resume work uniformly across surfaces.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Iterator

from ..config import (
    PROVIDER_FIELDS,
    SearchProviderStateConflict,
    commit_search_provider_config,
    config_has_llm_registry,
    load_context_budget_config,
    load_deep_checkpoint_settings_config,
    load_knowledge_graph_config,
    load_llm_registry,
    load_memory_settings_config,
    load_memory_dream_schedule_config,
    load_post_run_notes_config,
    load_retrieval_settings_config,
    load_search_usage_config,
    load_search_provider_state_config,
    merge_provider_config,
    normalize_memory_dream_schedule,
    registry_from_provider,
    save_context_budget_config,
    save_deep_checkpoint_settings_config,
    save_knowledge_graph_config,
    save_llm_registry,
    save_memory_settings_config,
    save_memory_dream_schedule_config,
    save_non_search_provider_config,
    save_post_run_notes_config,
    save_retrieval_settings_config,
    save_search_usage_config,
)
from ..connectivity import (
    check_llm_provider_connectivity,
    check_named_search_provider_connectivity,
    check_provider_models_reachable,
    check_search_provider_connectivity,
)
from ..model_listing import list_models
from ..observability import get_logger
from ..providers_registry import LLMProviderProfile, LLMRegistry, ModelProfile
from ..secret_vault import SecretVaultError
from ..settings import ProviderSettings, _configured_value
from ..standalone.llm import LLMClient
from ..standalone.search import SearchClient, SearchSettings
from .adapters import (
    BraveFirecrawlSearchAdapter,
    Document,
    LLMAdapter,
    OpenAICompatLLMAdapter,
    ResilientSearchAdapter,
    SearchToolAdapter,
)
from .budget import (
    COUNCIL_SIZE_DEFAULT,
    Mode,
    RunRequest,
    retrieval_budget_for,
    validate_deadline_budget,
)
from .context_budget import (
    ContextBudgetConfig,
    coerce_payload,
    default_config,
    from_env,
    recommend as recommend_context_budget_config,
    resolve_context_budget,
)
from .deep_checkpoint import (
    CheckpointAlreadyContinued,
    CheckpointCodecError,
    CheckpointStateConflict,
    decode_deep_checkpoint,
)
from .deep_checkpoint_settings import (
    DeepCheckpointSettings,
    coerce_deep_checkpoint_settings,
    default_deep_checkpoint_settings,
)
from .cost_estimate import RunCostEstimateRequest, estimate_run_cost
from .council import CouncilMemberSpec
from .content_sanitizer import sanitize_ingest_text
from .llm_call_params import LLMCallParams, resolve_llm_call_params
from .errors import BaselineError, ErrorCode
from .deadline import ProviderCallBudget, use_provider_call_budget
from .history import SCOPE_CROSS, SCOPE_INTRA, ConversationIndex
from .indexing import Indexer
from .job_backend import JobBackend, JobEnqueueStateUnknown, create_job_backend, redis_url
from .knowledge_graph import KnowledgeGraphService
from .knowledge_graph.job_events import TERMINAL_STATUSES, synthetic_terminal_event
from .knowledge_graph.jobs import create_wiki_map_job_backend
from .knowledge_graph.refresh_jobs import KnowledgeGraphRefreshQueue
from .knowledge_graph.settings import (
    KnowledgeGraphConfig,
    coerce_knowledge_graph_config,
    default_knowledge_graph_config,
)
from .knowledge_graph.wiki_generator import WikiGenerator
from .knowledge_graph.wiki_map import WikiMapBuilder
from .knowledge_graph.wiki_store import KnowledgeGraphWikiStore
from .library_index import LibraryIndexer
from .library_retrieval import LibraryRetriever
from .local_document_import import (
    MAX_UPLOAD_BYTES,
    parse_local_document,
    sanitize_import_filename,
    sanitize_import_title,
)
from .memory import MemoryEngine, MemoryLayer
from .memory_distillation import (
    MemoryDistillationCancelled,
    MemoryDistillationError,
    MemoryDistillationService,
)
from .memory_lifecycle import MemoryLifecycleService
from .memory_optimizer import MemoryOptimizer
from .memory_probe import run_memory_format_probe
from .memory_refresh_jobs import MemoryRefreshQueue
from .memory_settings import (
    MemorySettings,
    coerce_memory_settings,
    default_memory_settings,
    with_format_probe,
    with_stale_format_probe,
)
from .model_router import ModelRouter, OllamaLLMAdapter, build_model_router
from .orchestrator import DeepRunPaused, Orchestrator
from .post_run_notes_config import PostRunNotesConfig, coerce_post_run_notes_config, default_post_run_notes_config
from .retrieval import Retriever, build_reranker
from .retrieval_settings import RetrievalSettings, coerce_retrieval_settings, default_retrieval_settings
from .search_options import freeze_search_locale
from .search_usage import (
    DEFAULT_SEARCH_USAGE_POLICY,
    SearchUsagePolicy,
    normalize_search_usage_policy,
    provider_capabilities,
    resolve_search_usage,
)
from .search_provider_state import (
    SEARCH_PROVIDER_IDS,
    derive_provider_state,
    external_search_service_status,
    normalize_state_block,
    probe_can_persist_draft,
    probe_snapshot,
)
from .semantic_enrichment import (
    RQSemanticEnrichmentBackend,
    SemanticEnricher,
    SemanticEnrichmentOutcome,
    SemanticEnrichmentRequest,
    ThreadSemanticBackend,
)
from .snapshots import SnapshotWriter
from .store import DocumentWriteRejected, SessionStateConflict, Store
from .vector_backend import create_vector_backend

JsonDict = dict[str, Any]
_LOG = get_logger("deepresearch.service")

# Upper bound for free-text inputs (notes, enhance query). Guards LLM cost/DoS and
# unbounded SQLite growth; generous enough for any legitimate research prompt/note.
_MAX_TEXT_INPUT = 8000
_MAX_REPORT_INPUT = 200_000
_SNAPSHOT_PREVIEW_CHARS = 50_000
_DREAM_SCHEDULE_INTERVALS = {"daily": 86400.0, "weekly": 604800.0}
_DREAM_ACTIVE_STATUSES = {"queued", "running", "cancelling"}
_DREAM_TERMINAL_STATUSES = {"completed", "cancelled", "error"}
_DREAM_RECONCILE_GRACE_S = 60.0
_UPLOAD_REPORT_EXTENSIONS = {".md", ".txt"}
_SECRET_VALUE_RE = re.compile(r"(sk-|ghp_|AKIA|xoxb-|xoxp-|BEGIN .*PRIVATE KEY)", re.IGNORECASE)
_SEARCH_PROVIDER_FIELDS = frozenset(
    {
        "brave_api_key",
        "tavily_api_key",
        "exa_api_key",
        "firecrawl_api_key",
        "firecrawl_base_url",
    }
)
_SEARCH_PROVIDER_KEY_FIELDS = {
    "brave": "brave_api_key",
    "tavily": "tavily_api_key",
    "exa": "exa_api_key",
    "firecrawl": "firecrawl_api_key",
}

# Built-in council perspectives used when the LLM chair is unavailable/fails
# (graceful degradation). Distinct lenses so members argue from different angles.
_DEFAULT_COUNCIL_LENSES = (
    "supportive synthesizer building the strongest evidence-backed case",
    "skeptical risk analyst stress-testing every claim",
    "domain expert focused on technical accuracy and definitions",
    "pragmatic implementer weighing trade-offs and feasibility",
    "contrarian surfacing overlooked angles and counter-evidence",
)
_CHAIR_PROMPT = (
    "You are the council chair. For the question below, propose {n} DISTINCT analytical "
    "perspectives for {n} council members to argue from. One short perspective per line, "
    "no numbering.\n\nQuestion: {query}\n\nPerspectives:"
)

_VOLCENGINE_CODING_HINT = (
    "Volcengine coding endpoint (/api/coding/v3) is not OpenAI chat-compatible; "
    "use https://ark.cn-beijing.volces.com/api/v3 for DeepResearch."
)
def _volcengine_base_hint(base_url: str) -> str:
    """Return a config hint when a Volcengine coding base URL is detected."""

    if "/api/coding/" in str(base_url or "").lower():
        return _VOLCENGINE_CODING_HINT
    return ""


def _default_snapshot_root(db_path: str | Path) -> Path:
    """Return a stable local snapshot root for service-managed source snapshots."""

    if str(db_path) == ":memory:":
        return Path("data") / "snapshots"
    path = Path(db_path)
    parent = path.parent if path.parent != Path("") else Path(".")
    return parent / "snapshots"


def _default_chroma_root(db_path: str | Path) -> Path | None:
    """Return the derived Chroma cache directory for a SQLite database path."""

    if str(db_path) == ":memory:":
        return None
    path = Path(db_path)
    parent = path.parent if path.parent != Path("") else Path(".")
    return parent / "chroma"


def _safe_local_path(path: str | Path | None) -> str | None:
    """Return a user-facing local path without exposing parent directories."""

    if path is None:
        return None
    value = Path(path)
    parent = value.parent.name
    return f".../{parent}/{value.name}" if parent else value.name


def _safe_metadata(raw: Any) -> JsonDict:
    """Return JSON-like metadata without obvious secret keys or values."""

    if not isinstance(raw, dict):
        return {}
    out: JsonDict = {}
    for key, value in raw.items():
        name = str(key)
        if any(part in name.lower() for part in ("key", "token", "secret", "password")):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            text = str(value)
            if _SECRET_VALUE_RE.search(text):
                continue
            out[name] = value
    return out


def _report_title(query: str, mode: str) -> str:
    """Build a bounded report title from mode and query."""

    cleaned = re.sub(r"\s+", " ", str(query or "")).strip()
    if len(cleaned) > 80:
        cleaned = f"{cleaned[:77].rstrip()}..."
    prefix = "Council report" if mode == "council" else "Deep report"
    return f"{prefix}: {cleaned}" if cleaned else prefix


def _auto_note_content(answer: Any) -> str:
    """Create a deterministic post-run note without another LLM call."""

    summary = str(getattr(answer, "summary", "") or "").strip()
    text = summary or str(getattr(answer, "text", "") or "").strip()
    return text[:500].strip()


def _is_http_url(url: str) -> bool:
    """Return whether a URL should be considered live-fetchable."""

    lowered = str(url or "").strip().lower()
    return lowered.startswith("http://") or lowered.startswith("https://")


def _append_probe_hint(message: str, hint: str) -> str:
    """Append a non-secret probe hint without duplicating text."""

    text = str(message or "").strip()
    if not hint or hint in text:
        return text or hint
    return f"{text}; {hint}" if text else hint


class BaselineService:
    """High-level facade over the baseline subsystem."""

    def __init__(
        self,
        *,
        provider: ProviderSettings,
        db_path: str | Path = ":memory:",
        prefer_local_models: bool = False,
        enforce_grounding_gate: bool = True,
        db_encryption_key: str | None = None,
        config_path: str | Path | None = None,
        snapshot_root: str | Path | None = None,
        credential_sources: dict[str, str] | None = None,
    ) -> None:
        self._provider = provider
        self._provider_config_lock = threading.RLock()
        self._db_path = db_path
        self._prefer_local = prefer_local_models
        self._enforce_grounding_gate = enforce_grounding_gate
        self._config_path = Path(config_path) if config_path else None
        self._search_provider_sources = self._normalize_search_provider_sources(
            credential_sources
        )
        self._search_provider_state = (
            load_search_provider_state_config(self._config_path)
            if self._config_path is not None
            else normalize_state_block(None)
        )
        self._volatile_search_probes: dict[str, JsonDict] = {}
        self._search_usage_policy: SearchUsagePolicy = (
            load_search_usage_config(self._config_path)
            if self._config_path is not None
            else DEFAULT_SEARCH_USAGE_POLICY
        )
        self._search_usage_lock = threading.RLock()
        # Saved context-budget config (DR-ITER-057); None falls through to env/default.
        self._context_budget_config: ContextBudgetConfig | None = self._load_context_budget()
        self._memory_dream_schedule = self._load_memory_dream_schedule()
        self._post_run_notes_config: PostRunNotesConfig = self._load_post_run_notes_config()
        self._memory_settings: MemorySettings = self._load_memory_settings()
        self._knowledge_graph_config: KnowledgeGraphConfig = self._load_knowledge_graph_config()
        self._retrieval_settings: RetrievalSettings = self._load_retrieval_settings()
        self._deep_checkpoint_settings: DeepCheckpointSettings = (
            self._load_deep_checkpoint_settings()
        )
        self._registry: LLMRegistry | None = None
        self._test_memory_llm: LLMAdapter | None = None
        self._test_semantic_embedder: LLMAdapter | None = None
        self._memory_refresh_inline = False
        self._knowledge_graph_refresh_inline = False
        self._store = Store(db_path, encryption_key=db_encryption_key)
        self._snapshot_writer = SnapshotWriter(snapshot_root or _default_snapshot_root(db_path))
        try:
            with self._snapshot_writer.operation_lock():
                self._snapshot_orphan_sweep = self._snapshot_writer.sweep_orphans(
                    referenced_paths=set(self._store.list_snapshot_storage_paths()),
                    limit=4096,
                )
        except Exception:  # noqa: BLE001 - startup cleanup must not block the service
            self._snapshot_orphan_sweep = {
                "scanned": 0,
                "deleted": 0,
                "warnings": ["snapshot_sweep_failed"],
            }
        self._knowledge_graph = KnowledgeGraphService(
            self._store,
            on_rebuild_succeeded=self._on_knowledge_graph_rebuild_succeeded,
        )
        self._wiki_store = KnowledgeGraphWikiStore(self._store)
        self._wiki_map = WikiMapBuilder(self._store)
        self._wiki_job_backend = create_wiki_map_job_backend(job_fn=self._run_wiki_refresh_job_safe)
        # Job backend (thread default; rq for the Docker stack). Constructed once so an
        # rq misconfiguration (Redis unreachable) fails fast at service startup.
        self._job_backend: JobBackend = create_job_backend(
            run_fn=self._run_safe,
            memory_dream_fn=self.execute_memory_dream_job,
        )
        # Created lazily after the first committed continuity write. Keeping this
        # outside ``_wire_runtime`` prevents provider reconfiguration from replacing
        # a live bounded queue or losing pending work.
        self._semantic_backend_lock = threading.RLock()
        self._semantic_backend: (
            ThreadSemanticBackend | RQSemanticEnrichmentBackend | None
        ) = None
        self._semantic_shutdown = False
        self._wire_runtime()
        self._reconcile_memory_dream_runs()

    def _wire_runtime(self) -> None:
        """(Re)build the LLM router, search, retriever, indexer and orchestrator.

        Shared by ``__init__`` and runtime ``reconfigure`` so both paths produce an
        identical runtime from the current ``self._provider``. Never raises on an
        unconfigured provider; ``self._llm_ready`` reflects backend availability.
        """

        provider = self._provider
        self._store.set_provider_status(self._provider_status())
        self._registry = self._load_or_build_registry()

        # Default LLM: prefer the registry's default role binding, falling back to the
        # flat provider triple. The router keeps Ollama/LiteLLM failover semantics.
        eff_provider = self._effective_default_provider()
        llm_client = (
            LLMClient(base_url=eff_provider.llm_base_url, api_key=eff_provider.llm_api_key, model=eff_provider.llm_model)
            if eff_provider.llm_base_url and eff_provider.llm_api_key and eff_provider.llm_model
            else None
        )
        try:
            self._llm = build_model_router(eff_provider, llm_client=llm_client, prefer_local=self._prefer_local)
            self._llm_ready = True
        except BaselineError:
            self._llm = _UnconfiguredLLM()
            self._llm_ready = False

        search_client = None
        if any(
            (
                provider.brave_api_key,
                provider.firecrawl_api_key,
                provider.exa_api_key,
                provider.tavily_api_key,
            )
        ):
            search_client = SearchClient(
                SearchSettings(
                    brave_api_key=provider.brave_api_key,
                    firecrawl_api_key=provider.firecrawl_api_key,
                    firecrawl_base_url=provider.firecrawl_base_url or "https://api.firecrawl.dev",
                    exa_api_key=provider.exa_api_key,
                    tavily_api_key=provider.tavily_api_key,
                )
            )
        embedder = self._build_embedder(llm_client)
        # Reranker is env-selected (DR-ITER-025): embedding (default) | http cross-encoder | off.
        reranker = build_reranker(embedder)
        # Pluggable ANN index over the store (DR-ITER-020); graceful fallback to brute force.
        self._vector_backend = create_vector_backend(
            self._store,
            chroma_persist_enabled=self._effective_chroma_persist_enabled(),
            chroma_persist_path=self._chroma_persist_path(),
            chroma_ephemeral_forced=self._chroma_ephemeral_forced(),
        )
        retriever = Retriever(
            self._store,
            embedder=embedder,
            reranker=reranker,
            vector_backend=self._vector_backend,
        )
        self._indexer = Indexer(self._store, embedder=embedder)
        self._library_indexer = LibraryIndexer(self._store, indexer=self._indexer)
        self._library_retriever = LibraryRetriever(
            self._store,
            embedder=embedder,
            vector_backend=self._vector_backend,
        )
        self._search = (
            ResilientSearchAdapter(search_client, store=self._store, retriever=retriever)
            if search_client is not None
            else None
        )
        self._embedder = embedder
        # Single embedder source drives document/library indexing, memory, and
        # conversation-history semantic recall.
        self._memory = MemoryEngine(self._store, embedder=embedder)
        self._memory_lifecycle = MemoryLifecycleService(self._store, index_memory=self._index_memory_for_library)
        self._memory_optimizer = MemoryOptimizer(
            self._store,
            memory=self._memory,
            lifecycle=self._memory_lifecycle,
            index_memory=self._index_memory_for_library,
            memory_llm=self._resolve_memory_llm,
        )
        self._memory_distiller = MemoryDistillationService(
            store=self._store,
            memory_llm=self._resolve_memory_llm,
            memory_thinking_budget=self._resolve_memory_thinking_budget,
        )
        self._wiki_generator = WikiGenerator(
            store=self._store,
            memory_llm=self._resolve_memory_llm,
        )
        self._memory_refresh_queue = MemoryRefreshQueue(
            refresh_fn=self._run_memory_refresh_safe,
            run_inline=self._memory_refresh_inline,
        )
        self._knowledge_graph_refresh_queue = KnowledgeGraphRefreshQueue(
            refresh_fn=self._run_knowledge_graph_refresh_safe,
            run_inline=self._knowledge_graph_refresh_inline,
        )
        self._history = ConversationIndex(self._store, embedder=embedder)
        council_members = self._resolve_council_members(self._registry.roles.council)
        self._orchestrator = Orchestrator(
            store=self._store,
            llm=self._llm,
            retriever=retriever,
            indexer=self._indexer,
            search=self._search,
            memory=self._memory,
            council_members=council_members,
            enforce_grounding_gate=self._enforce_grounding_gate,
            llm_resolver=self._resolve_run_llm,
            council_resolver=self._resolve_run_council,
            params_resolver=self._resolve_run_llm_params,
            history=self._history,
            context_budget=self._effective_context_budget(),
            post_run_artifacts=self._persist_library_artifacts,
            post_continuity_persisted=self._enqueue_semantic_after_continuity,
            library_retriever=self._library_retriever,
        )

    # --- LLM registry resolution ---------------------------------------------

    def _load_or_build_registry(self) -> LLMRegistry:
        """Return the active LLM registry (explicit file registry, else synthesized)."""

        if self._config_path is not None:
            try:
                if config_has_llm_registry(self._config_path):
                    return load_llm_registry(self._config_path, environ=os.environ)
            except SecretVaultError:
                raise
            except Exception:  # noqa: BLE001 - never hide a failed migration or read
                raise SecretVaultError("Provider registry could not be loaded safely") from None
        if self._registry is not None:
            # Preserve in-memory registry mutations when there is no file to reload from.
            return self._registry
        return registry_from_provider(self._provider)

    def _effective_default_provider(self) -> ProviderSettings:
        """Return provider settings whose LLM triple reflects the registry default role."""

        ref = self._registry.roles.default
        resolved = self._registry.resolve(ref) if ref else None
        if resolved:
            profile, model = resolved
            if profile.enabled and profile.type != "ollama" and _configured_value(profile.base_url) and profile.has_key() and model:
                return replace(self._provider, llm_base_url=profile.base_url, llm_api_key=profile.api_key, llm_model=model)
        return self._provider

    def _build_embedder(self, default_client: LLMClient | None) -> LLMAdapter | None:
        """Return the embedder adapter (registry embed role, else the default client)."""

        ref = self._registry.roles.embed
        if ref:
            adapter = self._adapter_for_ref(ref)
            if adapter is not None:
                return adapter
        return OpenAICompatLLMAdapter(default_client) if default_client else None

    def _resolve_memory_llm(self) -> LLMAdapter | None:
        """Return the configured memory-processing model, with no implicit fallback."""

        if self._test_memory_llm is not None:
            return self._test_memory_llm
        ref = self._memory_provider_ref()
        return self._adapter_for_ref(ref) if ref else None

    def _memory_provider_ref(self) -> str:
        """Return the current memory role binding without exposing provider secrets."""

        return str(self._registry.roles.memory if self._registry is not None else "")

    def _resolve_memory_thinking_budget(self) -> int | None:
        """Return the effective Memory writer thinking budget for the current role."""

        if not self._memory_settings.memory_thinking_enabled:
            return 0
        ref = self._memory_provider_ref()
        resolved = self._registry.resolve(ref) if ref else None
        if not resolved:
            return None
        profile, model_id = resolved
        return resolve_llm_call_params(profile.get_model(model_id)).thinking_budget

    def _mark_memory_probe_stale(self, reason: str) -> None:
        """Mark cached Memory format probe stale and persist the safe settings block."""

        next_settings = with_stale_format_probe(self._memory_settings, reason=reason)
        if next_settings is self._memory_settings:
            return
        self._memory_settings = next_settings
        if self._config_path is not None:
            save_memory_settings_config(self._config_path, self._memory_settings)

    def _mark_memory_probe_stale_if_ref_changed(self, previous_ref: str, *, reason: str) -> None:
        if str(previous_ref or "") != self._memory_provider_ref():
            self._mark_memory_probe_stale(reason)

    @staticmethod
    def _provider_id_from_ref(ref: str) -> str:
        text = str(ref or "").strip()
        return text.split(":", 1)[0] if text else ""

    @staticmethod
    def _model_id_from_ref(ref: str) -> str:
        text = str(ref or "").strip()
        return text.split(":", 1)[1] if ":" in text else ""

    def _memory_ref_targets_provider(self, provider_id: str, ref: str | None = None) -> bool:
        return self._provider_id_from_ref(ref or self._memory_provider_ref()) == str(provider_id or "").strip()

    def _memory_ref_targets_model(self, provider_id: str, model_id: str, ref: str | None = None) -> bool:
        text = ref or self._memory_provider_ref()
        return self._memory_ref_targets_provider(provider_id, text) and self._model_id_from_ref(text) == str(model_id or "").strip()

    @staticmethod
    def _provider_probe_signature(profile: LLMProviderProfile | None) -> tuple[Any, ...]:
        if profile is None:
            return ()
        models = tuple(
            (
                model.id,
                model.kind,
                model.context_window,
                model.thinking_budget,
                model.temperature,
                model.dimensions,
                model.enabled,
                tuple(sorted(model.capabilities.to_config().items())),
            )
            for model in profile.models
        )
        return (profile.type, profile.base_url, profile.enabled, profile.has_key(), models)

    def _adapter_for_ref(self, ref: str) -> LLMAdapter | None:
        """Resolve a ``provider:model`` reference into a usable LLM adapter (or None)."""

        resolved = self._registry.resolve(ref)
        if not resolved:
            return None
        profile, model = resolved
        if not profile.enabled:
            return None
        if profile.type == "ollama":
            chosen = model or (profile.models[0].id if profile.models else "llama3.1")
            adapter = OllamaLLMAdapter(
                base_url=profile.base_url or "http://127.0.0.1:11434",
                model=chosen,
            )
            setattr(adapter, "_deepresearch_model_ref", ref)
            return adapter
        if not (_configured_value(profile.base_url) and profile.has_key() and model):
            return None
        adapter = OpenAICompatLLMAdapter(
            LLMClient(base_url=profile.base_url, api_key=profile.api_key, model=model),
            provider_type=profile.type,
            provider_id=profile.id,
        )
        setattr(adapter, "_deepresearch_model_ref", ref)
        return adapter

    def _resolve_council_members(self, refs: tuple[str, ...] | list[str]) -> list[LLMAdapter]:
        """Resolve a list of ``provider:model`` refs into distinct council adapters."""

        members: list[LLMAdapter] = []
        for ref in dict.fromkeys(refs):  # de-duplicate while preserving order
            adapter = self._adapter_for_ref(ref)
            if adapter is not None:
                members.append(adapter)
        return members

    def _resolve_run_llm(self, request: RunRequest) -> LLMAdapter | None:
        """Per-run default LLM override (``llm_model`` ref) or None to use the default."""

        if not request.llm_model:
            return None
        adapter = self._adapter_for_ref(request.llm_model)
        if adapter is None:
            return None
        # An explicit model selection still needs the Router's single attempt and
        # fallback owner.  Returning the raw adapter here previously let Composer
        # own a second stream-to-sync path and omitted attempts[] from the public
        # writer diagnostics observed by T52 Live.
        router = ModelRouter([adapter])
        setattr(router, "_deepresearch_model_ref", request.llm_model)
        setattr(router, "model", getattr(adapter, "model", None))
        return router

    def _resolve_run_llm_params(self, request: RunRequest) -> LLMCallParams:
        """Resolve effective sampling/thinking params: run override > model default > fallback."""

        ref = request.llm_model or self._registry.roles.default
        model_profile: ModelProfile | None = None
        resolved = self._registry.resolve(ref) if ref else None
        if resolved:
            profile, model_id = resolved
            model_profile = profile.get_model(model_id)
        return resolve_llm_call_params(
            model_profile,
            run_temperature=request.temperature,
            run_thinking_effort=request.thinking_effort,
        )

    def _resolve_run_council(self, request: RunRequest) -> list[CouncilMemberSpec]:
        """Council-chair assignment: ``council_size`` members from the candidate pool.

        The candidate pool is the per-run ``council_models`` (or the legacy
        ``council`` role refs); a single candidate model is replicated across
        members so one model can argue from several distinct perspectives. The
        chair assigns perspectives via the default LLM, falling back to built-in
        lenses on any failure (graceful degradation).
        """

        return self._assign_council(request)

    def _assign_council(self, request: RunRequest) -> list[CouncilMemberSpec]:
        """Build ``council_size`` member specs (model + perspective) for a run."""

        size = request.council_size or COUNCIL_SIZE_DEFAULT
        refs = request.council_models or list(self._registry.roles.council)
        adapters = self._resolve_council_members(refs)
        if not adapters:
            fallback = self._resolve_run_llm(request) or self._adapter_for_ref(self._registry.roles.default)
            if fallback is None and self._llm_ready:
                fallback = self._llm
            adapters = [fallback] if fallback is not None else []
        if not adapters:
            return []
        perspectives = self._council_perspectives(request.query, size)
        specs: list[CouncilMemberSpec] = []
        for index in range(size):
            adapter = adapters[index % len(adapters)]
            perspective = perspectives[index] if index < len(perspectives) else ""
            specs.append(CouncilMemberSpec(adapter=adapter, perspective=perspective, name=f"Member {index + 1}"))
        return specs

    def _council_perspectives(self, query: str, size: int) -> list[str]:
        """Return ``size`` distinct perspectives (LLM chair + built-in lens padding)."""

        out: list[str] = list(self._chair_perspectives(query, size))[:size]
        for lens in _DEFAULT_COUNCIL_LENSES:
            if len(out) >= size:
                break
            if lens not in out:
                out.append(lens)
        index = 0
        while len(out) < size:  # only when size exceeds the built-in lens count
            out.append(_DEFAULT_COUNCIL_LENSES[index % len(_DEFAULT_COUNCIL_LENSES)])
            index += 1
        return out[:size]

    def _chair_perspectives(self, query: str, size: int) -> list[str]:
        """Ask the default LLM (chair) for ``size`` perspectives; empty on any failure."""

        chair = self._adapter_for_ref(self._registry.roles.default)
        if chair is None and self._llm_ready:
            chair = self._llm
        if chair is None:
            return []
        prompt = _CHAIR_PROMPT.format(n=size, query=query[:500])
        try:
            result = chair.complete([{"role": "user", "content": prompt}], temperature=0.4, max_tokens=300)
        except Exception:  # noqa: BLE001 - perspectives are best-effort (graceful degradation)
            return []
        lenses: list[str] = []
        for line in result.text.splitlines():
            cleaned = re.sub(r"^[\s\-*\d.]+", "", line).strip()
            if len(cleaned) > 3:
                lenses.append(cleaned)
        return lenses[:size]

    def reconfigure(
        self,
        provider: ProviderSettings,
        *,
        credential_sources: dict[str, str] | None = None,
        reload_search_provider_state: bool = False,
    ) -> None:
        """Replace provider settings and rebuild the runtime in place."""

        with self._provider_config_lock:
            self._provider = provider
            if credential_sources is not None:
                self._search_provider_sources = self._normalize_search_provider_sources(
                    credential_sources
                )
            if reload_search_provider_state and self._config_path is not None:
                self._search_provider_state = load_search_provider_state_config(
                    self._config_path
                )
            self._wire_runtime()

    def _normalize_search_provider_sources(
        self,
        sources: dict[str, str] | None,
    ) -> dict[str, str]:
        """Return one authoritative credential source per live search provider."""

        status = self._provider.safe_status()
        key_fields = {
            "brave": "brave_api_key",
            "tavily": "tavily_api_key",
            "exa": "exa_api_key",
            "firecrawl": "firecrawl_api_key",
        }
        normalized: dict[str, str] = {}
        for provider, field in key_fields.items():
            if not status.get(field):
                normalized[provider] = "none"
                continue
            source = str((sources or {}).get(provider) or "config").strip().lower()
            normalized[provider] = source if source in {"config", "secure", "env"} else "config"
        return normalized

    def reload_memory_runtime_for_worker(self, provider: ProviderSettings) -> None:
        """Reload local provider/Memory config for a process-cached RQ worker."""

        self._provider = provider
        self._memory_settings = self._load_memory_settings()
        self._memory_dream_schedule = self._load_memory_dream_schedule()
        self._registry = None
        self._test_memory_llm = None
        self._wire_runtime()

    def reload_post_run_memory_settings_for_worker(self) -> None:
        """Reload the post-turn gate for a process-cached research worker."""

        latest = self._load_memory_settings()
        with self._provider_config_lock:
            self._memory_settings = latest

    def reload_semantic_runtime_for_worker(self, provider: ProviderSettings) -> None:
        """Reload only the explicit embed role for a semantic RQ work horse."""

        with self._provider_config_lock:
            self._provider = provider
            self._registry = None
            self._registry = self._load_or_build_registry()
            self._test_semantic_embedder = None

    def execute_semantic_enrichment_job(
        self,
        payload: JsonDict,
        *,
        research_busy: Callable[[], bool] | None = None,
        refresh_embedder_each_batch: bool = False,
    ) -> SemanticEnrichmentOutcome:
        """Synchronously enrich one active session without touching run state."""

        request = SemanticEnrichmentRequest.from_payload(payload)
        try:
            if refresh_embedder_each_batch:
                embedder = None
                embedder_resolver = self._resolve_semantic_embedder
            else:
                embedder = self._resolve_semantic_embedder()
                embedder_resolver = None
            outcome = SemanticEnricher(
                store=self._store,
                embedder=embedder,
                research_busy=research_busy,
                embedder_resolver=embedder_resolver,
            ).run(request)
        except Exception:  # noqa: BLE001 - maintenance cannot affect terminal runs
            outcome = SemanticEnrichmentOutcome(
                status="failed",
                session_id=request.session_id,
                run_id=request.run_id,
                trigger=request.trigger,
                reason_codes=("failed_internal",),
            )
        _LOG.info(
            "semantic_enrichment_finished",
            status=outcome.status,
            trigger=outcome.trigger,
            scanned_count=outcome.scanned_count,
            written_count=outcome.written_count,
            skipped_count=outcome.skipped_count,
            batch_count=outcome.batch_count,
            provider_attempts=outcome.provider_attempts,
            duration_ms=outcome.duration_ms,
            reason_codes=list(outcome.reason_codes),
        )
        return outcome

    def _semantic_embedder(self) -> LLMAdapter | None:
        """Resolve the explicit embed role; never fall back to a chat-only model."""

        if self._test_semantic_embedder is not None:
            return self._test_semantic_embedder
        if self._registry is None:
            self._registry = self._load_or_build_registry()
        ref = str(self._registry.roles.embed or "").strip()
        return self._adapter_for_ref(ref) if ref else None

    def _resolve_semantic_embedder(self) -> LLMAdapter | None:
        """Capture the current explicit embed role under the provider config lock."""

        with self._provider_config_lock:
            return self._semantic_embedder()

    def _enqueue_semantic_after_continuity(
        self,
        request: RunRequest,
        outcome: Any,
        run_id: str,
        persisted_rows: dict[str, list[str]],
    ) -> None:
        """Queue maintenance only after raw continuity rows have committed."""

        if not any(persisted_rows.values()) or not request.session_id:
            return
        backend = self._semantic_enqueue_backend()
        if backend is None:
            return
        accepted = backend.enqueue(
            SemanticEnrichmentRequest(
                session_id=request.session_id,
                run_id=str(run_id or ""),
                trigger="post_run",
            )
        )
        if accepted is False:
            _LOG.info(
                "semantic_enrichment_coalesced_or_full",
                trigger="post_run",
            )

    def _semantic_enqueue_backend(
        self,
    ) -> ThreadSemanticBackend | RQSemanticEnrichmentBackend | None:
        """Return the process-lifetime semantic backend without rebuilding it."""

        with self._semantic_backend_lock:
            if self._semantic_shutdown:
                return None
            if self._semantic_backend is not None:
                return self._semantic_backend
            if self._job_backend.name == "rq":
                try:
                    import redis as redis_lib
                except Exception as error:  # noqa: BLE001
                    raise RuntimeError(
                        "RQ semantic maintenance requires the existing redis package"
                    ) from error
                connection = redis_lib.Redis.from_url(
                    redis_url(),
                    socket_connect_timeout=2,
                    socket_timeout=2,
                    health_check_interval=30,
                )
                connection.ping()
                self._semantic_backend = RQSemanticEnrichmentBackend(connection)
            else:
                self._semantic_backend = ThreadSemanticBackend(
                    lambda semantic_request: self.execute_semantic_enrichment_job(
                        semantic_request.to_payload(),
                        refresh_embedder_each_batch=True,
                    )
                )
            return self._semantic_backend

    def shutdown(self) -> None:
        """Stop accepting in-process semantic work and join its sole consumer."""

        with self._semantic_backend_lock:
            if self._semantic_shutdown:
                return
            self._semantic_shutdown = True
            backend = self._semantic_backend
            self._semantic_backend = None
        if isinstance(backend, ThreadSemanticBackend):
            backend.shutdown(timeout_s=5.0)

    def reload_search_provider_state_for_worker(self) -> None:
        """Refresh secret-free External provider readiness without rebuilding adapters."""

        if self._config_path is None:
            return
        latest = load_search_provider_state_config(self._config_path)
        with self._provider_config_lock:
            self._search_provider_state = latest

    def enforce_worker_search_credential_snapshot(self, request: RunRequest) -> RunRequest:
        """Drop only changed config credentials from a frozen External path.

        Toggle/probe changes do not alter queued runs. A config-owned credential
        revision mismatch is different: the worker no longer possesses the
        credential authorized for that run, so that provider is skipped while
        fallback, local, Memory, cache, and indexed_external remain available.
        """

        usage = dict(request.resolved_search_usage or {})
        frozen = usage.get("credential_revision_snapshot")
        if not isinstance(frozen, dict):
            return request
        with self._provider_config_lock:
            current = self._search_provider_states()
        mismatched: set[str] = set()
        for provider, expected in frozen.items():
            state = current.get(str(provider))
            if not isinstance(state, dict) or state.get("credential_source") != "config":
                mismatched.add(str(provider))
                continue
            try:
                if int(state.get("credential_revision") or 0) != int(expected):
                    mismatched.add(str(provider))
            except (TypeError, ValueError):
                mismatched.add(str(provider))
        if not mismatched:
            return request
        usage["search_chain"] = [
            provider for provider in (usage.get("search_chain") or []) if provider not in mismatched
        ]
        usage["extract_chain"] = [
            provider for provider in (usage.get("extract_chain") or []) if provider not in mismatched
        ]
        reasons = [str(item) for item in (usage.get("reason_codes") or [])]
        reasons.extend(f"{provider}_provider_credential_changed" for provider in sorted(mismatched))
        usage["reason_codes"] = list(dict.fromkeys(reasons))
        usage["external_search_status"] = {
            "status": "warn" if usage["search_chain"] else "unavailable",
            "runtime_ready": bool(usage["search_chain"]),
            "reason_codes": list(dict.fromkeys(reasons)),
        }
        return replace(request, resolved_search_usage=usage)

    @property
    def store(self) -> Store:
        """Expose the underlying store (read-only intent)."""

        return self._store

    @property
    def orchestrator(self) -> Orchestrator:
        """Expose the orchestrator."""

        return self._orchestrator

    def start_knowledge_graph_rebuild(self, payload: JsonDict | None = None) -> JsonDict:
        """Start a manual knowledge graph rebuild."""

        return self._knowledge_graph.start_rebuild(payload or {})

    def get_knowledge_graph_rebuild_job(
        self,
        job_id: str,
        *,
        workspace_id: str = "default",
    ) -> JsonDict:
        """Return one rebuild job after workspace isolation."""

        return self._knowledge_graph.get_rebuild_job(
            job_id,
            workspace_id=workspace_id or "default",
        )

    def knowledge_graph_rebuild_now(self, payload: JsonDict | None = None, *, job_id: str | None = None) -> JsonDict:
        """Run a knowledge graph rebuild synchronously for workers/tests."""

        return self._knowledge_graph.rebuild_now(payload or {}, job_id=job_id)

    def mark_knowledge_graph_rebuild_failed(self, job_id: str, *, message: str) -> None:
        """Mark a queued/running knowledge graph rebuild failed."""

        self._knowledge_graph.mark_rebuild_failed(job_id, message=message)

    def get_knowledge_graph_status(self, workspace_id: str = "default") -> JsonDict:
        """Return active graph version and latest build status."""

        return self._knowledge_graph.get_status(workspace_id or "default")

    def get_knowledge_graph(
        self,
        *,
        workspace_id: str = "default",
        limit: int = 300,
        edge_types: list[str] | None = None,
    ) -> JsonDict:
        """Return a bounded active knowledge graph subgraph."""

        return self._knowledge_graph.get_graph(
            workspace_id=workspace_id or "default",
            limit=limit,
            edge_types=edge_types,
        )

    def get_knowledge_graph_session_map(
        self,
        *,
        workspace_id: str = "default",
        session_limit: int = 50,
        member_preview: int = 0,
    ) -> JsonDict:
        """Return the Session-centered active knowledge graph view."""

        return self._knowledge_graph.get_session_map(
            workspace_id=workspace_id or "default",
            session_limit=session_limit,
            member_preview=member_preview,
        )

    def get_knowledge_graph_wiki_map(self, *, workspace_id: str = "default") -> JsonDict:
        """Return the active Wiki Map view model without triggering LLM work."""

        return self._wiki_map.build(workspace_id=workspace_id or "default")

    def start_knowledge_graph_wiki_refresh(self, payload: JsonDict | None = None) -> JsonDict:
        """Start a manual Wiki Map refresh."""

        data = payload if isinstance(payload, dict) else {}
        workspace_id = str(data.get("workspace_id") or "default").strip() or "default"
        active_graph = self._knowledge_graph.get_status(workspace_id).get("active_version")
        graph_version_id = str((active_graph or {}).get("id") or "")
        self._wiki_store.fail_stale_orphan_refresh_jobs(workspace_id=workspace_id)
        running = self._wiki_store.get_running_refresh_job(workspace_id)
        if running:
            raise BaselineError(
                ErrorCode.KNOWLEDGE_GRAPH_WIKI_REFRESH_RUNNING,
                "Wiki Map refresh is already running",
                details={"job_id": running.get("id"), "workspace_id": workspace_id},
            )
        job = self._wiki_store.create_refresh_job(
            workspace_id=workspace_id,
            graph_version_id=graph_version_id,
            backend=self._wiki_job_backend.name,
        )
        job_id = str(job["id"])
        try:
            backend_job_id = self._wiki_job_backend.enqueue({"workspace_id": workspace_id}, job_id)
        except Exception as error:  # noqa: BLE001
            message = _safe_status_error(str(error)) or "Wiki Map refresh could not be queued. Please retry."
            self._wiki_store.mark_refresh_job_failed(
                job_id,
                error_code="ENQUEUE_FAILED",
                error_message=message,
            )
            raise BaselineError(
                ErrorCode.INTERNAL_ERROR,
                "Wiki Map refresh could not be queued",
                details={"job_id": job_id, "workspace_id": workspace_id, "error_code": "ENQUEUE_FAILED"},
            ) from error
        self._wiki_store.set_refresh_job_backend_id(job_id, backend_job_id)
        return self._wiki_store.get_refresh_job(job_id) or job

    def refresh_knowledge_graph_wiki_now(self, payload: JsonDict | None = None, *, job_id: str | None = None) -> JsonDict:
        """Run a Wiki Map refresh synchronously for workers/tests."""

        data = payload if isinstance(payload, dict) else {}
        workspace_id = str(data.get("workspace_id") or "default").strip() or "default"
        return self._wiki_generator.refresh(workspace_id=workspace_id, job_id=job_id)

    def mark_knowledge_graph_wiki_refresh_failed(self, job_id: str, *, message: str) -> None:
        """Mark a queued/running Wiki Map refresh failed."""

        self._wiki_store.mark_refresh_job_failed(
            _normalize_wiki_backend_job_id(job_id),
            error_code="INTERNAL_ERROR",
            error_message=message[:240],
        )

    def get_knowledge_graph_wiki_refresh_job(
        self,
        job_id: str,
        *,
        workspace_id: str = "default",
    ) -> JsonDict:
        """Return one Wiki Map refresh job."""

        job = self._wiki_store.get_refresh_job(str(job_id or ""))
        expected_workspace = str(workspace_id or "default").strip() or "default"
        if job is None or str(job.get("workspace_id") or "") != expected_workspace:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "Wiki Map refresh job not found",
                details={"job_id": str(job_id or "")},
            )
        return job

    def stream_knowledge_graph_rebuild_job(
        self,
        job_id: str,
        *,
        workspace_id: str = "default",
        after_seq: int = 0,
        poll_s: float = 0.2,
        timeout_s: float = 300.0,
        keepalive_s: float = 15.0,
    ) -> Iterator[bytes]:
        """Stream durable Graph rebuild events with cursor replay."""

        return self._stream_knowledge_graph_job(
            "graph",
            job_id,
            workspace_id=workspace_id,
            after_seq=after_seq,
            poll_s=poll_s,
            timeout_s=timeout_s,
            keepalive_s=keepalive_s,
        )

    def stream_knowledge_graph_wiki_refresh_job(
        self,
        job_id: str,
        *,
        workspace_id: str = "default",
        after_seq: int = 0,
        poll_s: float = 0.2,
        timeout_s: float = 300.0,
        keepalive_s: float = 15.0,
    ) -> Iterator[bytes]:
        """Stream durable Wiki refresh events with cursor replay."""

        return self._stream_knowledge_graph_job(
            "wiki",
            job_id,
            workspace_id=workspace_id,
            after_seq=after_seq,
            poll_s=poll_s,
            timeout_s=timeout_s,
            keepalive_s=keepalive_s,
        )

    def _stream_knowledge_graph_job(
        self,
        job_kind: str,
        job_id: str,
        *,
        workspace_id: str,
        after_seq: int,
        poll_s: float,
        timeout_s: float,
        keepalive_s: float,
    ) -> Iterator[bytes]:
        """Tail one workspace-owned maintenance job until an immutable terminal."""

        expected_workspace = str(workspace_id or "default").strip() or "default"
        clean_job_id = str(job_id or "").strip()
        last_seq = max(0, int(after_seq))
        idle_deadline = time.time() + max(1.0, float(timeout_s))
        next_keepalive = time.time() + max(1.0, float(keepalive_s))
        while time.time() < idle_deadline:
            job, events = self._knowledge_graph_job_state(
                job_kind,
                clean_job_id,
                workspace_id=expected_workspace,
                after_seq=last_seq,
            )
            if events:
                now = time.time()
                idle_deadline = now + max(1.0, float(timeout_s))
                next_keepalive = now + max(1.0, float(keepalive_s))
            for event in events:
                seq = int(event["seq"])
                if seq <= last_seq:
                    continue
                last_seq = seq
                yield _sse(
                    str(event["event_type"]),
                    dict(event["payload"]),
                    event_id=seq,
                )
                if str(event["event_type"]) == "job_terminal":
                    return
            if not events and str(job.get("status") or "") in TERMINAL_STATUSES:
                all_events = self._knowledge_graph_job_events(
                    job_kind,
                    clean_job_id,
                    workspace_id=expected_workspace,
                    after_seq=0,
                )
                if not all_events and last_seq < 1:
                    synthetic = synthetic_terminal_event(job_kind, job)
                    if synthetic is not None:
                        yield _sse(
                            str(synthetic["event_type"]),
                            dict(synthetic["payload"]),
                            event_id=int(synthetic["seq"]),
                        )
                return
            if not events and time.time() >= next_keepalive:
                yield b": keepalive\n\n"
                next_keepalive = time.time() + max(1.0, float(keepalive_s))
            time.sleep(max(0.01, float(poll_s)))

    def _knowledge_graph_job_state(
        self,
        job_kind: str,
        job_id: str,
        *,
        workspace_id: str,
        after_seq: int,
    ) -> tuple[JsonDict, list[JsonDict]]:
        if job_kind == "graph":
            job = self.get_knowledge_graph_rebuild_job(
                job_id,
                workspace_id=workspace_id,
            )
        else:
            job = self.get_knowledge_graph_wiki_refresh_job(
                job_id,
                workspace_id=workspace_id,
            )
        return job, self._knowledge_graph_job_events(
            job_kind,
            job_id,
            workspace_id=workspace_id,
            after_seq=after_seq,
        )

    def _knowledge_graph_job_events(
        self,
        job_kind: str,
        job_id: str,
        *,
        workspace_id: str,
        after_seq: int,
    ) -> list[JsonDict]:
        if job_kind == "graph":
            return self._knowledge_graph.list_rebuild_job_events(
                job_id,
                workspace_id=workspace_id,
                after_seq=after_seq,
            )
        self.get_knowledge_graph_wiki_refresh_job(
            job_id,
            workspace_id=workspace_id,
        )
        return self._wiki_store.list_refresh_job_events(
            job_id,
            after_seq=after_seq,
        )

    def get_knowledge_graph_node(self, node_id: str) -> JsonDict:
        """Return one knowledge graph node with provenance."""

        return self._knowledge_graph.get_node(node_id)

    def configure_runtime_for_tests(
        self,
        *,
        llm: LLMAdapter | None = None,
        memory_llm: LLMAdapter | None = None,
        semantic_embedder: LLMAdapter | None = None,
        memory_refresh_inline: bool | None = None,
        knowledge_graph_refresh_inline: bool | None = None,
        search: SearchToolAdapter | None = None,
        council_members: list[LLMAdapter] | None = None,
    ) -> None:
        """Install hermetic adapters for tests without touching provider config."""

        self._test_memory_llm = memory_llm
        self._test_semantic_embedder = semantic_embedder
        if memory_refresh_inline is not None:
            self._memory_refresh_inline = bool(memory_refresh_inline)
        if knowledge_graph_refresh_inline is not None:
            self._knowledge_graph_refresh_inline = bool(knowledge_graph_refresh_inline)
        if llm is not None:
            self._llm = llm
            self._llm_ready = True
        self._search = search
        embedder = self._llm if hasattr(self._llm, "embed") else self._embedder
        self._vector_backend = create_vector_backend(
            self._store,
            chroma_persist_enabled=self._effective_chroma_persist_enabled(),
            chroma_persist_path=self._chroma_persist_path(),
            chroma_ephemeral_forced=self._chroma_ephemeral_forced(),
        )
        retriever = Retriever(
            self._store,
            embedder=embedder,
            reranker=build_reranker(embedder),
            vector_backend=self._vector_backend,
        )
        self._indexer = Indexer(self._store, embedder=embedder)
        self._library_indexer = LibraryIndexer(self._store, indexer=self._indexer)
        self._library_retriever = LibraryRetriever(
            self._store,
            embedder=embedder,
            vector_backend=self._vector_backend,
        )
        self._memory = MemoryEngine(self._store, embedder=embedder)
        self._memory_lifecycle = MemoryLifecycleService(self._store, index_memory=self._index_memory_for_library)
        self._memory_optimizer = MemoryOptimizer(
            self._store,
            memory=self._memory,
            lifecycle=self._memory_lifecycle,
            index_memory=self._index_memory_for_library,
            memory_llm=self._resolve_memory_llm,
        )
        self._memory_distiller = MemoryDistillationService(
            store=self._store,
            memory_llm=self._resolve_memory_llm,
            memory_thinking_budget=self._resolve_memory_thinking_budget,
        )
        self._wiki_generator = WikiGenerator(
            store=self._store,
            memory_llm=self._resolve_memory_llm,
        )
        self._memory_refresh_queue = MemoryRefreshQueue(
            refresh_fn=self._run_memory_refresh_safe,
            run_inline=self._memory_refresh_inline,
        )
        self._knowledge_graph_refresh_queue = KnowledgeGraphRefreshQueue(
            refresh_fn=self._run_knowledge_graph_refresh_safe,
            run_inline=self._knowledge_graph_refresh_inline,
        )
        self._history = ConversationIndex(self._store, embedder=embedder)
        self._orchestrator = Orchestrator(
            store=self._store,
            llm=self._llm,
            retriever=retriever,
            indexer=self._indexer,
            search=search,
            memory=self._memory,
            council_members=council_members or [],
            enforce_grounding_gate=self._enforce_grounding_gate,
            history=self._history,
            context_budget=self._effective_context_budget(),
            post_run_artifacts=self._persist_library_artifacts,
            post_continuity_persisted=self._enqueue_semantic_after_continuity,
            library_retriever=self._library_retriever,
        )

    # --- context budget (DR-ITER-057) -----------------------------------------

    def _load_context_budget(self) -> ContextBudgetConfig | None:
        """Load the saved context-budget config from config.json (or None)."""

        if self._config_path is None:
            return None
        try:
            return load_context_budget_config(self._config_path)
        except Exception:  # noqa: BLE001 - a bad config block never blocks startup
            return None

    def _load_memory_dream_schedule(self) -> str:
        """Load the saved Dreaming schedule, defaulting to off."""

        if self._config_path is None:
            return "off"
        try:
            return load_memory_dream_schedule_config(self._config_path)
        except Exception:  # noqa: BLE001 - a bad config block never blocks startup
            return "off"

    def _load_post_run_notes_config(self) -> PostRunNotesConfig:
        """Load saved post-run notes config, defaulting to all modes disabled."""

        if self._config_path is None:
            return default_post_run_notes_config()
        try:
            return load_post_run_notes_config(self._config_path)
        except Exception:  # noqa: BLE001 - a bad config block never blocks startup
            return default_post_run_notes_config()

    def _load_memory_settings(self) -> MemorySettings:
        """Load saved Memory settings, defaulting to conservative built-ins."""

        if self._config_path is None:
            return default_memory_settings()
        try:
            return load_memory_settings_config(self._config_path)
        except Exception:  # noqa: BLE001 - a bad config block never blocks startup
            return default_memory_settings()

    def _load_knowledge_graph_config(self) -> KnowledgeGraphConfig:
        """Load saved Knowledge Graph settings, defaulting to automatic rebuild on."""

        if self._config_path is None:
            return default_knowledge_graph_config()
        try:
            return load_knowledge_graph_config(self._config_path)
        except Exception:  # noqa: BLE001 - a bad config block never blocks startup
            return default_knowledge_graph_config()

    def _load_retrieval_settings(self) -> RetrievalSettings:
        """Load retrieval settings, defaulting Chroma persistence to on."""

        if self._config_path is None:
            return default_retrieval_settings()
        try:
            return load_retrieval_settings_config(self._config_path)
        except Exception:  # noqa: BLE001 - a bad config block never blocks startup
            return default_retrieval_settings()

    def _load_deep_checkpoint_settings(self) -> DeepCheckpointSettings:
        """Load Deep checkpoint settings, defaulting automatic checkpoints to off."""

        if self._config_path is None:
            return default_deep_checkpoint_settings()
        try:
            return load_deep_checkpoint_settings_config(self._config_path)
        except Exception:  # noqa: BLE001 - a malformed optional block never blocks startup
            return default_deep_checkpoint_settings()

    def _effective_context_budget(self) -> ContextBudgetConfig:
        """Resolve the effective config (saved -> env -> default), no run override."""

        return resolve_context_budget(None, self._context_budget_config, os.environ)

    def _chroma_ephemeral_forced(self) -> bool:
        """Return whether env or in-memory DB forces Chroma into ephemeral mode."""

        return str(self._db_path) == ":memory:" or str(os.environ.get("DEEPRESEARCH_CHROMA_EPHEMERAL") or "").strip() == "1"

    def _chroma_persist_path(self) -> Path | None:
        """Return the effective disk path for derived Chroma cache data."""

        override = str(os.environ.get("DEEPRESEARCH_CHROMA_PERSIST_PATH") or "").strip()
        if override:
            return Path(override)
        return _default_chroma_root(self._db_path)

    def _effective_chroma_persist_enabled(self) -> bool:
        """Return whether Chroma disk persistence should be used by this runtime."""

        return bool(
            self._retrieval_settings.chroma_persist_enabled
            and not self._chroma_ephemeral_forced()
            and self._chroma_persist_path() is not None
        )

    def _retrieval_settings_payload(self, *, restart_required: bool = False) -> JsonDict:
        """Return saved and effective retrieval settings for API/UI clients."""

        effective_enabled = self._effective_chroma_persist_enabled()
        return {
            "saved": self._retrieval_settings.to_dict(),
            "effective": {
                "chroma_persist_enabled": effective_enabled,
                "chroma_persist_mode": "disk" if effective_enabled else "ephemeral",
                "chroma_persist_path": _safe_local_path(self._chroma_persist_path()) if effective_enabled else None,
                "ephemeral_forced": self._chroma_ephemeral_forced(),
                "kg_aware_retrieval_enabled": bool(
                    self._retrieval_settings.kg_aware_retrieval_enabled
                ),
            },
            "restart_required": restart_required,
        }

    def get_post_run_notes_config(self) -> JsonDict:
        """Return the active post-run notes config."""

        return self._post_run_notes_config.to_dict()

    def update_post_run_notes_config(self, payload: JsonDict) -> JsonDict:
        """Persist a post-run notes config update."""

        try:
            config = coerce_post_run_notes_config(payload)
        except ValueError as error:
            raise BaselineError(ErrorCode.INVALID_REQUEST, str(error)) from error
        self._post_run_notes_config = config
        if self._config_path is not None:
            save_post_run_notes_config(self._config_path, config)
        return config.to_dict()

    def get_memory_settings(self) -> JsonDict:
        """Return Memory continuity settings for Settings UI/API clients."""

        return self._memory_settings.to_dict()

    def update_memory_settings(self, payload: JsonDict) -> JsonDict:
        """Persist Memory continuity settings while preserving cached probe metadata."""

        if not isinstance(payload, dict):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "memory settings payload must be an object")
        current = self._memory_settings.to_dict()
        merged = {**current, **payload}
        if "format_probe" not in payload:
            merged["format_probe"] = current.get("format_probe")
        try:
            config = coerce_memory_settings(merged)
        except ValueError as error:
            raise BaselineError(ErrorCode.INVALID_REQUEST, str(error)) from error
        self._memory_settings = config
        if self._config_path is not None:
            save_memory_settings_config(self._config_path, config)
        return config.to_dict()

    def probe_memory_format(self) -> JsonDict:
        """Run a synthetic, secret-free structured-output probe for the Memory role."""

        probe = run_memory_format_probe(
            self._resolve_memory_llm(),
            provider_ref=self._memory_provider_ref(),
        )
        self._memory_settings = with_format_probe(self._memory_settings, probe)
        if self._config_path is not None:
            save_memory_settings_config(self._config_path, self._memory_settings)
        return probe

    def get_knowledge_graph_config(self) -> JsonDict:
        """Return the active Knowledge Graph settings."""

        return self._knowledge_graph_config.to_dict()

    def update_knowledge_graph_config(self, payload: JsonDict) -> JsonDict:
        """Persist Knowledge Graph settings."""

        try:
            config = coerce_knowledge_graph_config(payload)
        except ValueError as error:
            raise BaselineError(ErrorCode.INVALID_REQUEST, str(error)) from error
        self._knowledge_graph_config = config
        if self._config_path is not None:
            save_knowledge_graph_config(self._config_path, config)
        return config.to_dict()

    def get_retrieval_settings(self) -> JsonDict:
        """Return saved and effective retrieval settings."""

        return self._retrieval_settings_payload()

    def update_retrieval_settings(self, payload: JsonDict) -> JsonDict:
        """Persist retrieval settings; restart is required for vector backend rewiring."""

        if not isinstance(payload, dict):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "retrieval settings payload must be an object")
        current = self._retrieval_settings.to_dict()
        previous_chroma_enabled = self._retrieval_settings.chroma_persist_enabled
        try:
            config = coerce_retrieval_settings({**current, **payload})
        except ValueError as error:
            raise BaselineError(ErrorCode.INVALID_REQUEST, str(error)) from error
        self._retrieval_settings = config
        if self._config_path is not None:
            save_retrieval_settings_config(self._config_path, config)
        return self._retrieval_settings_payload(
            restart_required=(
                config.chroma_persist_enabled != previous_chroma_enabled
            )
        )

    def get_deep_checkpoint_settings(self) -> JsonDict:
        """Return the global Deep auto-checkpoint setting."""

        return self._deep_checkpoint_settings.to_dict()

    def update_deep_checkpoint_settings(self, payload: JsonDict) -> JsonDict:
        """Validate and persist the global Deep auto-checkpoint setting."""

        try:
            config = coerce_deep_checkpoint_settings(payload)
        except ValueError as error:
            raise BaselineError(ErrorCode.INVALID_REQUEST, str(error)) from error
        self._deep_checkpoint_settings = config
        if self._config_path is not None:
            save_deep_checkpoint_settings_config(self._config_path, config)
        return config.to_dict()

    def purge_cache(self, scope: str = "all") -> JsonDict:
        """Delete retrieval cache entries for an operator-selected scope."""

        normalized = str(scope or "all")
        if normalized not in {"all", "search", "fetch"}:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "scope must be all, search, or fetch")
        return {"purged_count": self._store.cache_purge(scope=normalized), "scope": normalized}

    def get_context_budget(self) -> JsonDict:
        """Return the effective config plus env/default context for the Settings UI."""

        env_config = from_env(os.environ)
        return {
            "effective": self._effective_context_budget().to_dict(),
            "saved": self._context_budget_config.to_dict() if self._context_budget_config else None,
            "env": env_config.to_dict() if env_config else None,
            "default": default_config().to_dict(),
            "source": (
                "saved" if self._context_budget_config is not None
                else "env" if env_config is not None
                else "default"
            ),
        }

    def update_context_budget(self, payload: JsonDict) -> JsonDict:
        """Persist a global context-budget config; updates memory + config.json.

        Validates the payload (``INVALID_REQUEST`` on bad shape) so a later run
        never has to. The in-memory copy is updated synchronously so the very next
        ``create_run`` in this process resolves against the new config.
        """

        try:
            config = coerce_payload(payload)
        except ValueError as error:
            raise BaselineError(ErrorCode.INVALID_REQUEST, str(error)) from error
        self._context_budget_config = config
        if self._config_path is not None:
            save_context_budget_config(self._config_path, config)
        return self.get_context_budget()

    def recommend_context_budget(self, payload: JsonDict | None = None) -> JsonDict:
        """Recommend a preset from runtime signals (heuristic; no LLM)."""

        payload = payload or {}
        session_id = str(payload.get("session_id") or "").strip()
        memory_policy = str(payload.get("memory_policy") or "off").lower()
        workspace_key = str(payload.get("workspace_key") or "default")
        turn_count = 0
        if session_id:
            try:
                turn_count = len(self._store.list_messages(session_id))
            except Exception:  # noqa: BLE001 - missing session => no turn signal
                turn_count = 0
        memory_count = self._memory.count(workspace_key) if self._memory is not None else 0
        return recommend_context_budget_config(
            embed_ready=self._embedder is not None,
            workspace_memory_count=memory_count,
            session_turn_count=turn_count,
            memory_policy=memory_policy,
        )

    def get_search_usage(self) -> JsonDict:
        """Return the normalized global search usage policy."""

        with self._search_usage_lock:
            return self._search_usage_policy.to_dict()

    def update_search_usage(self, payload: JsonDict | None) -> JsonDict:
        """Validate and persist the complete global search usage policy."""

        try:
            policy = normalize_search_usage_policy(payload, require_complete=True)
        except ValueError as error:
            raise BaselineError(ErrorCode.INVALID_REQUEST, str(error)) from error
        with self._search_usage_lock:
            if self._config_path is not None:
                save_search_usage_config(self._config_path, policy)
            self._search_usage_policy = policy
        return policy.to_dict()

    def _search_provider_states(self, *, now: float | None = None) -> dict[str, JsonDict]:
        """Return live External-provider states; internal retrieval is out of scope."""

        status = self._provider.safe_status()
        key_fields = {
            "brave": "brave_api_key",
            "tavily": "tavily_api_key",
            "exa": "exa_api_key",
            "firecrawl": "firecrawl_api_key",
        }
        saved = self._search_provider_state.get("providers")
        saved = saved if isinstance(saved, dict) else {}
        states: dict[str, JsonDict] = {}
        for provider in SEARCH_PROVIDER_IDS:
            state = derive_provider_state(
                provider,
                configured=bool(status.get(key_fields[provider])),
                credential_source=self._search_provider_sources.get(provider, "none"),
                saved=saved.get(provider) if isinstance(saved.get(provider), dict) else None,
                volatile_probe=self._volatile_search_probes.get(provider),
                now=now,
            ).to_dict()
            states[provider] = state
        return states

    def _configured_search_providers(self) -> dict[str, JsonDict]:
        """Return provider-state inputs for deterministic External path resolution."""

        return self._search_provider_states()

    # --- runs -----------------------------------------------------------------

    def create_run(
        self,
        payload: JsonDict,
        *,
        background: bool = True,
        _trusted_run_metadata: JsonDict | None = None,
    ) -> JsonDict:
        """Create and start a run; returns ``{run_id, status}``."""

        requested_at = time.time()
        payload = dict(payload or {})
        update_session_policy = "search_usage_policy" in payload
        requested_session_policy: SearchUsagePolicy | None = None
        if update_session_policy and payload.get("search_usage_policy") is not None:
            try:
                requested_session_policy = normalize_search_usage_policy(
                    payload.get("search_usage_policy"),
                    require_complete=True,
                )
            except ValueError as error:
                raise BaselineError(ErrorCode.INVALID_REQUEST, str(error)) from error
        with self._search_usage_lock:
            global_search_usage = self._search_usage_policy
        if not payload.get("mode"):
            default_mode = self._session_mode_default(str(payload.get("session_id") or ""))
            if default_mode:
                payload["mode"] = default_mode
        request = RunRequest.from_payload(payload)
        resolved_run_budget = request.resolved_budget()
        validate_deadline_budget(resolved_run_budget, retrieval_budget_for(request.mode))
        if not self._llm_ready:
            raise BaselineError(
                ErrorCode.PROVIDER_NOT_CONFIGURED,
                "No LLM backend configured: set BYOK provider keys or start a local Ollama server.",
            )
        # Resolve the effective context budget in the API process and embed it into
        # the request so the (possibly out-of-process) worker uses it verbatim --
        # Settings changes take effect without a worker restart (DR-ITER-057).
        effective_budget = resolve_context_budget(
            request.context_budget, self._context_budget_config, os.environ
        )
        search_settings = getattr(getattr(self._search, "_client", None), "settings", SearchSettings())
        search_options, requested_locale, effective_locale = freeze_search_locale(
            request.search_options,
            search_settings,
        )
        existing_session = None
        if request.session_id:
            existing_session = self._store.get_session(request.session_id)
            if existing_session is None:
                raise BaselineError(
                    ErrorCode.SESSION_NOT_FOUND,
                    "session not found",
                    details={"session_id": request.session_id},
                )
            if existing_session.get("archived_at") is not None:
                raise BaselineError(
                    ErrorCode.INVALID_REQUEST,
                    "archived sessions cannot start runs",
                    details={"session_id": request.session_id},
                )
        kg_retrieval_requested = bool(
            self._retrieval_settings.kg_aware_retrieval_enabled
        )
        kg_workspace_id = str(
            (existing_session or {}).get("workspace")
            or request.workspace_key
            or "default"
        )
        kg_retrieval_snapshot = self._knowledge_graph.freeze_retrieval_snapshot(
            workspace_id=kg_workspace_id,
            requested=kg_retrieval_requested,
            mode=request.mode.value,
        )
        request = replace(
            request,
            context_budget=effective_budget.to_dict(),
            search_options=search_options,
            requested_locale=requested_locale,
            effective_locale=effective_locale,
            requested_at=requested_at,
            deadline_at=requested_at + float(resolved_run_budget.timeout_s),
            auto_checkpoint_enabled=(
                self._deep_checkpoint_settings.auto_checkpoint_enabled
                if request.mode is Mode.DEEP
                else False
            ),
            kg_aware_retrieval_enabled=kg_retrieval_requested,
            kg_retrieval_snapshot=kg_retrieval_snapshot,
        )

        def _freeze_search_usage(session_row: JsonDict | None) -> RunRequest:
            session_policy = (
                requested_session_policy
                if update_session_policy
                else _session_search_usage_policy(session_row)
            )
            effective_policy = session_policy or global_search_usage
            with self._provider_config_lock:
                provider_states = self._configured_search_providers()
                provider_config_revision = int(
                    self._search_provider_state.get("revision") or 0
                )
            resolved_usage = resolve_search_usage(
                effective_policy,
                source="session" if session_policy is not None else "global",
                configured=provider_states,
                requested_locale=requested_locale,
                provider_config_revision=provider_config_revision,
            )
            return replace(request, resolved_search_usage=resolved_usage.to_dict())

        def _run_metadata(frozen_request: RunRequest) -> JsonDict:
            trusted_metadata = dict(_trusted_run_metadata or {})
            regeneration = trusted_metadata.get("regeneration_context")
            if isinstance(regeneration, dict):
                regeneration = dict(regeneration)
                search_chain = list(
                    frozen_request.resolved_search_usage.get("search_chain") or []
                )
                regeneration["fresh_search_available"] = bool(search_chain)
                regeneration["fresh_search_provider_chain"] = search_chain
                regeneration["provider_config_revision"] = (
                    frozen_request.resolved_search_usage.get("provider_config_revision")
                )
                if search_chain:
                    regeneration.pop("diagnostic", None)
                else:
                    regeneration["diagnostic"] = "external_search_unavailable"
                trusted_metadata["regeneration_context"] = regeneration
            return {
                **(
                    {"search_options": frozen_request.search_options.to_dict()}
                    if frozen_request.search_options
                    else {}
                ),
                "requested_locale": dict(frozen_request.requested_locale),
                "effective_locale": dict(frozen_request.effective_locale),
                "resolved_search_usage": dict(frozen_request.resolved_search_usage),
                "requested_at": frozen_request.requested_at,
                "deadline_at": frozen_request.deadline_at,
                "auto_checkpoint_enabled": frozen_request.auto_checkpoint_enabled,
                "kg_aware_retrieval_enabled": (
                    frozen_request.kg_aware_retrieval_enabled
                ),
                "kg_retrieval_snapshot": dict(
                    frozen_request.kg_retrieval_snapshot
                ),
                # This snapshot contains only RunRequest fields (model references,
                # policy, and bounds); provider credentials are never part of it.
                "run_request_snapshot": frozen_request.to_payload(),
                **trusted_metadata,
                # Background workers must not claim this run until queue acceptance
                # is confirmed after enqueue returns. Inline execution is already
                # acknowledged by the current process.
                "enqueue_confirmed": not background,
            }

        rollback_previous_metadata: str | None = None
        rollback_previous_updated_at: float | None = None
        rollback_expected_policy_revision: int | None = None
        if request.session_id:
            session_id = request.session_id
            for attempt in range(5):
                request = _freeze_search_usage(existing_session)
                try:
                    created = self._store.update_session_and_create_run(
                        session_id,
                        mode=request.mode.value,
                        query=request.query,
                        budget=request.resolved_budget().to_dict(),
                        run_metadata=_run_metadata(request),
                        search_usage_policy=(
                            requested_session_policy.to_dict()
                            if requested_session_policy is not None
                            else None
                        ),
                        update_search_usage_policy=update_session_policy,
                        expected_session_metadata=(
                            str(existing_session.get("metadata") or "")
                            if not update_session_policy and existing_session is not None
                            else None
                        ),
                        expected_session_updated_at=(
                            float(existing_session.get("updated_at") or 0.0)
                            if not update_session_policy and existing_session is not None
                            else None
                        ),
                    )
                except SessionStateConflict:
                    existing_session = self._store.get_session(session_id)
                    if existing_session is None or existing_session.get("archived_at") is not None:
                        created = None
                        break
                    if attempt == 4:
                        raise BaselineError(
                            ErrorCode.INTERNAL_ERROR,
                            "session changed repeatedly while freezing search usage",
                            details={"session_id": session_id},
                        ) from None
                    continue
                break
            if created is None:
                raise BaselineError(
                    ErrorCode.SESSION_NOT_FOUND,
                    "session not found",
                    details={"session_id": session_id},
                )
            run_id = str(created["run_id"])
            rollback_previous_metadata = json.dumps(
                created.get("previous_session_metadata") or {}
            )
            rollback_previous_updated_at = float(
                created.get("previous_session_updated_at") or 0.0
            )
            rollback_expected_policy_revision = int(
                created.get("applied_search_usage_policy_revision") or 0
            )
        else:
            request = _freeze_search_usage(None)
            session_metadata = (
                {"search_usage_policy": requested_session_policy.to_dict()}
                if requested_session_policy is not None
                else {}
            )
            session_id, run_id = self._store.create_session_and_run(
                title=request.query[:80],
                mode=request.mode.value,
                query=request.query,
                budget=request.resolved_budget().to_dict(),
                session_metadata=session_metadata,
                run_metadata=_run_metadata(request),
            )
        request = replace(request, session_id=session_id)
        if background:
            try:
                self._job_backend.enqueue(request, run_id)
                if not self._store.confirm_run_enqueued(run_id):
                    raise JobEnqueueStateUnknown(
                        "research queue acceptance could not be committed"
                    )
            except JobEnqueueStateUnknown:
                self._store.abort_unconfirmed_enqueue(
                    run_id,
                    session_id,
                    previous_session_metadata=rollback_previous_metadata,
                    previous_session_updated_at=rollback_previous_updated_at,
                    restore_search_usage_policy=bool(request.session_id and update_session_policy),
                    expected_search_usage_policy_revision=rollback_expected_policy_revision,
                )
                raise
            except Exception:
                self._store.rollback_unenqueued_run(
                    run_id,
                    session_id,
                    created_session=existing_session is None,
                    previous_session_metadata=rollback_previous_metadata,
                    previous_session_updated_at=rollback_previous_updated_at,
                    restore_search_usage_policy=bool(request.session_id and update_session_policy),
                    expected_search_usage_policy_revision=rollback_expected_policy_revision,
                )
                raise
        else:
            self._run_safe(request, run_id)
        return {"run_id": run_id, "session_id": session_id, "status": "running"}

    def regenerate_run(self, run_id: str, *, background: bool = True) -> JsonDict:
        """Create a new run from one report plus a server-trusted evidence snapshot."""

        parent = self._store.get_run(run_id)
        if parent is None:
            raise BaselineError(
                ErrorCode.RUN_NOT_FOUND,
                "run not found",
                details={"run_id": run_id},
            )
        session_id = str(parent.get("session_id") or "")
        self._assert_session_accessible(session_id)
        if str(parent.get("status") or "") not in {
            "completed",
            "completed_degraded",
            "partial_timeout",
        } or not isinstance(parent.get("result"), dict):
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "only a delivered report can be regenerated",
                details={"run_id": run_id, "status": parent.get("status")},
            )

        metadata = parent.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        stored_snapshot = metadata.get("run_request_snapshot")
        payload = dict(stored_snapshot) if isinstance(stored_snapshot, dict) else {}
        payload.update(
            {
                "query": str(parent.get("query") or ""),
                "mode": str(parent.get("mode") or Mode.FAST.value),
                "session_id": session_id,
                # A regenerated run receives a fresh absolute deadline.
                "requested_at": 0.0,
                "deadline_at": 0.0,
            }
        )
        if "budget" not in payload and isinstance(parent.get("budget"), dict):
            payload["budget"] = dict(parent["budget"])

        evidence: list[JsonDict] = []
        for citation in self._store.list_citations(run_id)[:64]:
            if str(citation.get("source_status") or "active") != "active":
                continue
            artifact_type = str(citation.get("artifact_type") or "web_document")
            artifact_id = str(citation.get("artifact_id") or citation.get("document_id") or "")
            if artifact_type != "web_document" and not artifact_id:
                continue
            evidence.append(
                {
                    # Stored citations do not normally retain the source chunk PK.
                    # Zero is an explicit unbound sentinel; inventing a positive
                    # id could alias an unrelated chunk in the current database.
                    "chunk_id": int(citation.get("chunk_id") or 0),
                    **{
                        key: citation.get(key)
                        for key in (
                            "id",
                            "document_id",
                            "url",
                            "title",
                            "snippet",
                            "score",
                            "origin",
                            "artifact_type",
                            "artifact_id",
                            "grounding_eligible",
                            "provenance_refs",
                            "source_status",
                            "published_at",
                            "updated_at",
                            "effective_at",
                            "retrieved_at",
                        )
                    },
                }
            )
        trusted_metadata = {
            "regeneration_context": {
                "parent_run_id": run_id,
                "seed_evidence": evidence,
                "fresh_search_required": True,
                "created_at": time.time(),
            }
        }
        created = self.create_run(
            payload,
            background=background,
            _trusted_run_metadata=trusted_metadata,
        )
        return {**created, "parent_run_id": run_id}

    def acknowledge_run_confidence(self, run_id: str) -> JsonDict:
        """Record that the user accepted one report's low-confidence warning."""

        run = self._store.get_run(run_id)
        if run is None:
            raise BaselineError(
                ErrorCode.RUN_NOT_FOUND,
                "run not found",
                details={"run_id": run_id},
            )
        self._assert_session_accessible(str(run.get("session_id") or ""))
        acknowledged = self._store.acknowledge_run_confidence(run_id)
        if acknowledged is None:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "this report has no completed low-confidence warning to acknowledge",
                details={"run_id": run_id},
            )
        return acknowledged

    def estimate_run_cost(self, payload: JsonDict | None = None) -> JsonDict:
        """Return a pre-submit cost estimate without creating or running a run."""

        request = RunCostEstimateRequest.from_payload(payload or {})
        provider_type, model_id = self._cost_estimate_model_context(request)
        estimate = estimate_run_cost(
            request,
            model_id=model_id,
            provider_type=provider_type,
            search_provider=self._cost_estimate_search_provider(),
        )
        return estimate.to_dict()

    def _cost_estimate_model_context(self, request: RunCostEstimateRequest) -> tuple[str, str]:
        """Return non-secret provider/model identifiers for cost lookup."""

        ref = request.llm_model or self._registry.roles.default
        resolved = self._registry.resolve(ref) if ref else None
        if resolved:
            profile, model_id = resolved
            return profile.type, model_id
        if self._provider.llm_model:
            return "openai", self._provider.llm_model
        return "", ""

    def _cost_estimate_search_provider(self) -> str:
        """Return the configured search pricing family without exposing keys."""

        brave = _configured_value(self._provider.brave_api_key)
        firecrawl = _configured_value(self._provider.firecrawl_api_key)
        if _configured_value(self._provider.exa_api_key) or _configured_value(self._provider.tavily_api_key):
            return "mixed_provider_unknown"
        if brave and firecrawl:
            return "brave_firecrawl"
        if brave:
            return "brave"
        if firecrawl:
            return "firecrawl"
        return "none"

    def execute_run(self, request: RunRequest, run_id: str) -> None:
        """Execute an already-persisted run synchronously (used by the RQ worker)."""

        self._run_safe(request, run_id)

    def _run_safe(self, request: RunRequest, run_id: str) -> None:
        if not self._await_enqueue_confirmation(request, run_id):
            return
        try:
            self._orchestrator.run(request, run_id=run_id)
        except DeepRunPaused:
            pass  # paused checkpoint and terminal event were committed atomically
        except BaselineError:
            pass  # error already persisted to the run + run_events
        except Exception:  # noqa: BLE001 - never crash the worker thread
            pass

    def _await_enqueue_confirmation(self, request: RunRequest, run_id: str) -> bool:
        """Wait briefly for the API's durable queue-acceptance acknowledgement.

        A worker may start before ``enqueue()`` returns to the API, especially with
        the in-process thread backend. It must not claim the run or call a provider
        in that window. Legacy runs without the marker are treated as confirmed.
        """

        remaining = (
            max(0.0, float(request.deadline_at) - time.time())
            if request.deadline_at > 0
            else 5.0
        )
        wait_deadline = time.monotonic() + min(5.0, remaining)
        while True:
            state = self._store.enqueue_confirmation_state(run_id)
            if state == "confirmed":
                return True
            if state == "cancelling":
                # The API can only accept a cancellation after create_run returned,
                # which means queue acceptance was already confirmed. Let a fresh RQ
                # process enter Orchestrator.run so it can persist the cancelled
                # terminal state without calling any provider.
                return True
            if state != "pending":
                return False
            if time.monotonic() >= wait_deadline:
                # Resolve the final worker/API race in one SQLite transaction.
                # The worker must not terminalize here: only the API owns the
                # previous session snapshot needed to compensate an override.
                return self._store.expire_enqueue_confirmation_wait(run_id) == "confirmed"
            time.sleep(0.01)

    def fail_run(self, run_id: str, message: str, *, code: ErrorCode = ErrorCode.INTERNAL_ERROR) -> None:
        """Mark a run as failed out-of-band (e.g. the RQ work horse was killed).

        Used by the worker failure callback when the orchestrator never got the
        chance to persist a terminal state itself (timeout/OOM/crash). No-op if
        the run is already terminal so a late callback can never rewrite history.
        """

        run = self._store.get_run(run_id)
        if not run or str(run.get("status") or "") in {
            "completed",
            "completed_degraded",
            "cancelled",
            "error",
            "partial_timeout",
        }:
            return
        payload = BaselineError(code, message).to_payload()
        self._persist_stream_metrics_from_events(run_id)
        run = self._store.get_run(run_id) or run
        diagnostics = self._out_of_band_run_diagnostics(run, code=code)
        if code is ErrorCode.RUN_TIMEOUT and run.get("result"):
            if not self._store.transition_run(
                run_id,
                expected_statuses={"queued", "running", "cancelling"},
                status="partial_timeout",
                metadata_patch=diagnostics,
                events=[
                    (
                        "workflow_end",
                        {"status": "partial_timeout", "code": ErrorCode.RUN_TIMEOUT.value},
                    )
                ],
            ):
                return
            return
        generation_id = self._active_answer_generation(run_id)
        terminal_events: list[tuple[str, dict[str, object]]] = []
        if generation_id is not None:
            terminal_events.append(
                (
                    "answer_reset",
                    {
                        "generation_id": generation_id,
                        "reason": (
                            "deadline_exceeded"
                            if code is ErrorCode.RUN_TIMEOUT
                            else "draft_invalidated"
                        ),
                    },
                )
            )
        terminal_events.append(("error", payload))
        if not self._store.transition_run(
            run_id,
            expected_statuses={"queued", "running", "cancelling"},
            status="error",
            error=payload,
            metadata_patch=diagnostics,
            events=terminal_events,
        ):
            return

    @staticmethod
    def _out_of_band_run_diagnostics(run: dict[str, object], *, code: ErrorCode) -> dict[str, object]:
        """Complete T49 diagnostics when an RQ horse dies outside the orchestrator."""

        metadata = run.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, dict) else {}
        try:
            mode = Mode(str(run.get("mode") or Mode.FAST.value))
        except ValueError:
            mode = Mode.FAST
        retrieval = retrieval_budget_for(mode)
        retrieval_payload = metadata.get("retrieval_budget")
        retrieval_payload = (
            dict(retrieval_payload)
            if isinstance(retrieval_payload, dict)
            else retrieval.to_dict()
        )
        now = time.time()
        requested_at = float(metadata.get("requested_at") or run.get("created_at") or now)
        deadline_at = float(metadata.get("deadline_at") or 0.0)
        existing_summary = metadata.get("deadline_summary")
        existing_summary = dict(existing_summary) if isinstance(existing_summary, dict) else {}
        worker_started_at = float(
            existing_summary.get("worker_started_at") or run.get("updated_at") or now
        )
        phase_metrics = metadata.get("phase_metrics")
        phase_metrics = list(phase_metrics) if isinstance(phase_metrics, list) else []
        remaining_ms = (
            max(0.0, (deadline_at - now) * 1000.0)
            if deadline_at > 0
            else 0.0
        )
        phase_metrics.append(
            {
                "stage": "worker_failure_callback",
                "attempt": 0,
                "started_at": now,
                "finished_at": now,
                "duration_ms": 0.0,
                "status": "timeout" if code is ErrorCode.RUN_TIMEOUT else "failed",
                "deadline_remaining_before_ms": remaining_ms,
                "deadline_remaining_after_ms": remaining_ms,
            }
        )
        deadline_summary = {
            **existing_summary,
            "requested_at": requested_at,
            "deadline_at": deadline_at,
            "worker_started_at": worker_started_at,
            "writer_started_at": existing_summary.get("writer_started_at"),
            "terminal_at": now,
            "queue_delay_ms": max(0.0, (worker_started_at - requested_at) * 1000.0),
            "writer_reserve_s": retrieval.writer_reserve_s,
            "finalize_reserve_s": retrieval.finalize_reserve_s,
            "cleanup_reserve_s": retrieval.cleanup_reserve_s,
            "deadline_met": False,
            "cleanup_overrun_ms": (
                max(0.0, (now - deadline_at) * 1000.0) if deadline_at > 0 else 0.0
            ),
        }
        return {
            "phase_metrics": phase_metrics,
            "deadline_summary": deadline_summary,
            "retrieval_budget": retrieval_payload,
        }

    def _active_answer_generation(self, run_id: str) -> str | None:
        """Return the currently visible draft generation, if any."""

        active: str | None = None
        for event in self._store.list_events(run_id, after_seq=0):
            payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
            generation_id = str(payload.get("generation_id") or "") or None
            if event.get("type") == "answer_delta" and generation_id:
                active = generation_id
            elif event.get("type") == "answer_reset" and generation_id == active:
                active = None
            elif event.get("type") == "workflow_end":
                active = None
        return active

    def _persist_stream_metrics_from_events(self, run_id: str) -> None:
        """Recover bounded stream metrics when a worker dies outside the orchestrator."""

        run = self._store.get_run(run_id) or {}
        metadata = run.get("metadata") if isinstance(run.get("metadata"), dict) else {}
        existing = metadata.get("stream_metrics") if isinstance(metadata.get("stream_metrics"), dict) else {}
        deltas = [
            event
            for event in self._store.list_events(run_id, after_seq=0)
            if event.get("type") == "answer_delta"
        ]
        if not deltas:
            return
        delta_payloads = [
            event.get("payload") if isinstance(event.get("payload"), dict) else {}
            for event in deltas
        ]
        first_delta_ms = existing.get("first_answer_delta_ms")
        try:
            requested_at = float(metadata.get("requested_at") or 0.0)
            first_created_at = float(deltas[0].get("created_at") or 0.0)
        except (TypeError, ValueError):
            requested_at = 0.0
            first_created_at = 0.0
        if first_delta_ms is None and requested_at > 0 and first_created_at > 0:
            first_delta_ms = max(0.0, (first_created_at - requested_at) * 1000.0)
        metrics = {
            "first_answer_delta_ms": first_delta_ms,
            "delta_event_count": len(deltas),
            "delta_bytes": sum(
                len(str(payload.get("delta") or "").encode("utf-8"))
                for payload in delta_payloads
            ),
            "sync_fallback_used": bool(existing.get("sync_fallback_used", False)),
        }
        self._store.update_run(run_id, metadata_patch={"stream_metrics": metrics})

    def job_backend_status(self) -> JsonDict:
        """Return the active job backend + redis reachability for ``/health``."""

        return self._job_backend.health()

    def vector_backend_status(self) -> JsonDict:
        """Return the active vector-search backend for ``/health`` (DR-ITER-020)."""

        backend = getattr(self, "_vector_backend", None)
        if backend is None:
            status: JsonDict = {"vector_backend": None}
        else:
            status = backend.health()
        effective = self._retrieval_settings_payload()["effective"]
        status.setdefault("chroma_persist_enabled", effective["chroma_persist_enabled"])
        status.setdefault("chroma_persist_mode", effective["chroma_persist_mode"])
        status.setdefault("chroma_persist_path", effective["chroma_persist_path"])
        return status

    def get_run(self, run_id: str) -> JsonDict:
        """Return a run record or raise RUN_NOT_FOUND."""

        run = self._store.get_run(run_id)
        if not run:
            raise BaselineError(ErrorCode.RUN_NOT_FOUND, "run not found", details={"run_id": run_id})
        self._assert_session_accessible(str(run.get("session_id") or ""))
        if self._expire_stale_run_verification(run_id, run):
            run = self._store.get_run(run_id) or run
        run["citations"] = self._store.list_citations(run_id)
        return run

    def _expire_stale_run_verification(self, run_id: str, run: JsonDict) -> bool:
        """Turn an abandoned verifier into a visible low-confidence result."""

        if str(run.get("status") or "") not in {
            "completed",
            "completed_degraded",
            "partial_timeout",
        }:
            return False
        result = run.get("result")
        result = dict(result) if isinstance(result, dict) else {}
        verification = result.get("verification")
        verification = dict(verification) if isinstance(verification, dict) else {}
        if str(verification.get("status") or "") not in {"pending", "running"}:
            return False
        now = time.time()
        try:
            started_at = float(verification.get("started_at") or run.get("updated_at") or 0.0)
        except (TypeError, ValueError):
            started_at = 0.0
        try:
            report_updated_at = float(run.get("updated_at") or 0.0)
        except (TypeError, ValueError):
            report_updated_at = 0.0
        verification_age_anchor = max(started_at, report_updated_at)
        stale = verification_age_anchor > 0 and now >= verification_age_anchor + 60.0
        if not stale:
            return False
        failed = {
            **verification,
            "status": "failed",
            "completed_at": now,
            "reason_codes": ["verification_interrupted"],
        }
        return self._store.update_run_verification(
            run_id,
            verification=failed,
            confidence_label="uncertain",
            confidence_warning={
                "reasons": ["verification_interrupted"],
                "acknowledged": False,
            },
        )

    def pause_run(self, run_id: str) -> JsonDict:
        """Persist a cooperative pause request for a running Deep run."""

        run = self._store.get_run(run_id)
        if run is None:
            raise BaselineError(
                ErrorCode.RUN_NOT_FOUND,
                "run not found",
                details={"run_id": run_id},
            )
        self._assert_session_accessible(str(run.get("session_id") or ""))
        metadata = run.get("metadata") if isinstance(run.get("metadata"), dict) else {}
        if bool(metadata.get("deep_checkpoint_pause_fenced")):
            raise BaselineError(
                ErrorCode.RUN_NOT_PAUSABLE,
                "the final writer has started and this run can no longer be paused",
                details={"run_id": run_id, "status": run.get("status")},
            )
        result = self._store.request_run_pause(run_id)
        if result.get("accepted"):
            return result
        if result.get("status") == "not_found":
            raise BaselineError(
                ErrorCode.RUN_NOT_FOUND,
                "run not found",
                details={"run_id": run_id},
            )
        raise BaselineError(
            ErrorCode.RUN_NOT_PAUSABLE,
            "run is not at a pausable Deep state",
            details={
                "run_id": run_id,
                "status": result.get("status"),
                "reason": result.get("reason"),
            },
        )

    def get_run_checkpoint(self, run_id: str) -> JsonDict:
        """Return integrity and resume readiness without exposing checkpoint payload."""

        run = self._store.get_run(run_id)
        if run is None:
            raise BaselineError(
                ErrorCode.RUN_NOT_FOUND,
                "run not found",
                details={"run_id": run_id},
            )
        checkpoint = self._store.get_deep_checkpoint(run_id, include_payload=True)
        if checkpoint is None:
            raise BaselineError(
                ErrorCode.CHECKPOINT_NOT_FOUND,
                "checkpoint not found",
                details={"run_id": run_id},
            )
        return self._checkpoint_resume_view(run, checkpoint)

    def continue_run_from_checkpoint(
        self,
        run_id: str,
        payload: JsonDict,
        *,
        background: bool = True,
    ) -> JsonDict:
        """Claim a paused checkpoint and enqueue its sole child run."""

        if not isinstance(payload, dict):
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "continue payload must be an object",
            )
        try:
            expected_revision = int(payload.get("expected_checkpoint_revision"))
        except (TypeError, ValueError) as error:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "expected_checkpoint_revision is required",
            ) from error
        idempotency_key = str(payload.get("idempotency_key") or "").strip()
        if not idempotency_key or len(idempotency_key) > 200:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "idempotency_key must be a bounded non-empty string",
            )
        action = str(payload.get("action") or "continue").strip().lower()
        if action not in {"continue", "revise"}:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "action must be continue or revise",
            )
        remaining_plan = payload.get("remaining_plan")
        if action == "revise" and not isinstance(remaining_plan, dict):
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "remaining_plan is required for revise",
            )
        if action == "continue" and remaining_plan is not None:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "remaining_plan is only valid for revise",
            )
        if isinstance(remaining_plan, dict):
            if set(remaining_plan) != {"sections"} or not isinstance(
                remaining_plan.get("sections"), list
            ):
                raise BaselineError(
                    ErrorCode.INVALID_REQUEST,
                    "remaining_plan must contain only a sections array",
                )
            revised_sections = [
                " ".join(str(item or "").split())
                for item in remaining_plan["sections"]
            ]
            if (
                not revised_sections
                or len(revised_sections) > 32
                or any(not item or len(item) > 500 for item in revised_sections)
            ):
                raise BaselineError(
                    ErrorCode.INVALID_REQUEST,
                    "remaining_plan sections must contain 1 to 32 bounded titles",
                )
            remaining_plan = {"sections": revised_sections}
        incremental_budget = payload.get("incremental_budget")
        if incremental_budget is not None and not isinstance(incremental_budget, dict):
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "incremental_budget must be an object",
            )
        parent = self._store.get_run(run_id)
        if parent is None:
            raise BaselineError(
                ErrorCode.RUN_NOT_FOUND,
                "run not found",
                details={"run_id": run_id},
            )
        checkpoint = self._store.get_deep_checkpoint(run_id, include_payload=True)
        if checkpoint is None:
            raise BaselineError(
                ErrorCode.CHECKPOINT_NOT_FOUND,
                "checkpoint not found",
                details={"run_id": run_id},
            )
        replay = self._store.get_deep_checkpoint_claim(
            run_id,
            idempotency_key=idempotency_key,
        )
        if replay is not None:
            return {
                **replay,
                "parent_run_id": run_id,
                "idempotent_replay": True,
            }
        view = self._checkpoint_resume_view(parent, checkpoint)
        if view["integrity"] != "valid":
            raise BaselineError(
                ErrorCode.CHECKPOINT_INVALID,
                "checkpoint failed integrity validation",
                details={"run_id": run_id},
            )
        if view["resume_readiness"] not in {"ready", "degraded"}:
            raise BaselineError(
                ErrorCode.CHECKPOINT_STATE_CONFLICT,
                "checkpoint is not resumable",
                details={
                    "run_id": run_id,
                    "resume_readiness": view["resume_readiness"],
                    "reason_codes": view["reason_codes"],
                },
            )
        if not self._llm_ready:
            raise BaselineError(
                ErrorCode.PROVIDER_NOT_CONFIGURED,
                "No LLM backend configured: set BYOK provider keys or start a local Ollama server.",
            )
        checkpoint_payload = checkpoint.get("payload")
        if not isinstance(checkpoint_payload, dict):
            raise BaselineError(
                ErrorCode.CHECKPOINT_INVALID,
                "checkpoint payload is unavailable",
            )
        request_payload = checkpoint_payload.get("request")
        if not isinstance(request_payload, dict):
            raise BaselineError(
                ErrorCode.CHECKPOINT_INVALID,
                "checkpoint request snapshot is unavailable",
            )
        try:
            base_request = RunRequest.from_payload(request_payload)
        except BaselineError as error:
            raise BaselineError(
                ErrorCode.CHECKPOINT_INVALID,
                "checkpoint request snapshot is invalid",
            ) from error
        if base_request.mode is not Mode.DEEP:
            raise BaselineError(
                ErrorCode.CHECKPOINT_INVALID,
                "only Deep checkpoints can be continued",
            )
        overrides = dict(base_request.overrides)
        if incremental_budget is not None:
            unknown_budget_fields = set(incremental_budget) - set(
                base_request.resolved_budget().to_dict()
            )
            if unknown_budget_fields:
                raise BaselineError(
                    ErrorCode.INVALID_REQUEST,
                    "incremental_budget contains unknown fields",
                )
            overrides.update(incremental_budget)
        requested_at = time.time()
        request = replace(
            base_request,
            session_id=str(parent.get("session_id") or ""),
            requested_at=requested_at,
            deadline_at=0.0,
            checkpoint_parent_run_id=run_id,
            checkpoint_revision=expected_revision,
            lineage_id=str(checkpoint.get("lineage_id") or ""),
            checkpoint_action=action,
            checkpoint_remaining_plan=(
                dict(remaining_plan) if isinstance(remaining_plan, dict) else None
            ),
            checkpoint_incremental_budget=(
                dict(incremental_budget)
                if isinstance(incremental_budget, dict)
                else None
            ),
            overrides=overrides,
        )
        resolved_budget = request.resolved_budget()
        validate_deadline_budget(resolved_budget, retrieval_budget_for(Mode.DEEP))
        request = replace(
            request,
            deadline_at=requested_at + float(resolved_budget.timeout_s),
        )
        metadata = {
            "requested_at": request.requested_at,
            "deadline_at": request.deadline_at,
            "requested_locale": dict(request.requested_locale),
            "effective_locale": dict(request.effective_locale),
            "resolved_search_usage": dict(request.resolved_search_usage),
            "auto_checkpoint_enabled": request.auto_checkpoint_enabled,
            "kg_aware_retrieval_enabled": request.kg_aware_retrieval_enabled,
            "kg_retrieval_snapshot": dict(request.kg_retrieval_snapshot),
            "checkpoint_parent_run_id": run_id,
            "checkpoint_revision": expected_revision,
            "lineage_id": request.lineage_id,
            "checkpoint_action": action,
            "checkpoint_resume_readiness": view["resume_readiness"],
            "enqueue_confirmed": not background,
        }
        try:
            claimed = self._store.claim_deep_checkpoint_and_create_child(
                run_id=run_id,
                expected_revision=expected_revision,
                idempotency_key=idempotency_key,
                mode=Mode.DEEP.value,
                query=request.query,
                budget=resolved_budget.to_dict(),
                metadata=metadata,
            )
        except CheckpointAlreadyContinued as error:
            raise BaselineError(
                ErrorCode.CHECKPOINT_ALREADY_CONTINUED,
                "checkpoint already continued",
                details={"run_id": run_id},
            ) from error
        except (CheckpointStateConflict, CheckpointCodecError) as error:
            raise BaselineError(
                ErrorCode.CHECKPOINT_STATE_CONFLICT,
                "checkpoint changed before it could be continued",
                details={"run_id": run_id},
            ) from error
        child_run_id = str(claimed["run_id"])
        if claimed.get("idempotent_replay"):
            return {
                "run_id": child_run_id,
                "session_id": claimed["session_id"],
                "status": claimed["status"],
                "parent_run_id": run_id,
                "idempotent_replay": True,
            }
        request = replace(request, session_id=str(claimed["session_id"]))
        try:
            if background:
                self._job_backend.enqueue(request, child_run_id)
                if not self._store.confirm_run_enqueued(child_run_id):
                    raise JobEnqueueStateUnknown(
                        "checkpoint child queue acceptance could not be committed"
                    )
            else:
                self._run_safe(request, child_run_id)
        except Exception:
            self._store.rollback_deep_checkpoint_child_claim(
                run_id=run_id,
                child_run_id=child_run_id,
                idempotency_key=idempotency_key,
            )
            raise
        return {
            "run_id": child_run_id,
            "session_id": request.session_id,
            "status": "running",
            "parent_run_id": run_id,
            "idempotent_replay": False,
        }

    def abandon_run_checkpoint(
        self,
        run_id: str,
        payload: JsonDict | None = None,
    ) -> JsonDict:
        """Retain a checkpoint for audit but make it permanently non-resumable."""

        if self._store.get_run(run_id) is None:
            raise BaselineError(
                ErrorCode.RUN_NOT_FOUND,
                "run not found",
                details={"run_id": run_id},
            )
        expected = (payload or {}).get("expected_checkpoint_revision")
        try:
            expected_revision = int(expected) if expected is not None else None
        except (TypeError, ValueError) as error:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "expected_checkpoint_revision must be an integer",
            ) from error
        try:
            checkpoint = self._store.abandon_deep_checkpoint(
                run_id,
                expected_revision=expected_revision,
            )
        except CheckpointAlreadyContinued as error:
            raise BaselineError(
                ErrorCode.CHECKPOINT_ALREADY_CONTINUED,
                "checkpoint already continued",
                details={"run_id": run_id},
            ) from error
        except CheckpointStateConflict as error:
            raise BaselineError(
                ErrorCode.CHECKPOINT_STATE_CONFLICT,
                "checkpoint revision changed",
                details={"run_id": run_id},
            ) from error
        if checkpoint is None:
            raise BaselineError(
                ErrorCode.CHECKPOINT_NOT_FOUND,
                "checkpoint not found",
                details={"run_id": run_id},
            )
        return {
            "run_id": run_id,
            "revision": checkpoint["revision"],
            "disposition": checkpoint["disposition"],
            "resume_readiness": "unavailable",
        }

    def _checkpoint_resume_view(
        self,
        run: JsonDict,
        checkpoint: JsonDict,
    ) -> JsonDict:
        """Separate stored integrity from current-environment resume readiness."""

        integrity = str(checkpoint.get("integrity") or "invalid")
        reason_codes: list[str] = []
        readiness = "ready"
        disposition = str(checkpoint.get("disposition") or "")
        if integrity != "valid":
            readiness = "blocked"
            reason_codes.append("checkpoint_integrity_invalid")
        elif disposition in {"claimed", "abandoned"}:
            readiness = "unavailable"
            reason_codes.append(f"checkpoint_{disposition}")
        elif str(run.get("status") or "") != "paused":
            readiness = "blocked"
            reason_codes.append("parent_not_paused")
        session = self._store.get_session(str(run.get("session_id") or ""))
        if session is None:
            readiness = "blocked"
            reason_codes.append("session_missing")
        elif session.get("archived_at") is not None:
            readiness = "blocked"
            reason_codes.append("session_archived")
        checkpoint_payload = checkpoint.get("payload")
        if integrity == "valid" and isinstance(checkpoint_payload, dict):
            for evidence in checkpoint_payload.get("evidence_snapshot") or []:
                if not isinstance(evidence, dict):
                    continue
                artifact_type = str(evidence.get("artifact_type") or "")
                artifact_id = str(
                    evidence.get("artifact_id")
                    or evidence.get("document_id")
                    or ""
                )
                if (
                    artifact_type == "library_document"
                    and artifact_id
                    and self._store.get_document(artifact_id) is None
                ):
                    reason_codes.append("source_tombstone")
                if (
                    artifact_type == "memory"
                    and artifact_id
                    and self._store.get_memory(artifact_id) is None
                ):
                    reason_codes.append("memory_missing")
            request_snapshot = checkpoint_payload.get("request")
            if isinstance(request_snapshot, dict):
                resolved = request_snapshot.get("resolved_search_usage")
                resolved = resolved if isinstance(resolved, dict) else {}
                frozen_revision = int(resolved.get("provider_config_revision") or 0)
                current_revision = int(self._search_provider_state.get("revision") or 0)
                if frozen_revision and frozen_revision != current_revision:
                    reason_codes.append("provider_config_drift")
        if readiness == "ready" and reason_codes:
            readiness = "degraded"
        return {
            "run_id": str(run.get("id") or checkpoint.get("run_id") or ""),
            "lineage_id": checkpoint.get("lineage_id"),
            "revision": checkpoint.get("revision"),
            "schema_version": checkpoint.get("schema_version"),
            "stage": checkpoint.get("stage"),
            "disposition": disposition,
            "claimed_child_run_id": checkpoint.get("claimed_child_run_id"),
            "created_at": checkpoint.get("created_at"),
            "updated_at": checkpoint.get("updated_at"),
            "integrity": integrity,
            "resume_readiness": readiness,
            "reason_codes": list(dict.fromkeys(reason_codes)),
            "budget_usage": (
                dict(checkpoint_payload.get("budget_usage") or {})
                if isinstance(checkpoint_payload, dict)
                else {}
            ),
        }

    def cancel_run(self, run_id: str) -> JsonDict:
        """Request cooperative cancellation of a run."""

        if not self._store.get_run(run_id):
            raise BaselineError(ErrorCode.RUN_NOT_FOUND, "run not found", details={"run_id": run_id})
        self._orchestrator.request_cancel(run_id)
        return {"run_id": run_id, "status": "cancelling"}

    def stream_run(
        self,
        run_id: str,
        *,
        after_seq: int = 0,
        poll_s: float = 0.2,
        timeout_s: float = 300.0,
        keepalive_s: float = 15.0,
    ) -> Iterator[bytes]:
        """Yield SSE bytes for a run, replaying persisted events then tailing new ones.

        ``timeout_s`` is an *idle* timeout: it resets whenever a new event arrives, so
        long multi-stage runs are never cut off mid-stream just for being long
        (DR-ITER-058). While idle, SSE comment keepalives (``: keepalive``) are sent
        every ``keepalive_s`` so proxies don't drop the connection. If the run row
        reaches a terminal status without a terminal event in ``run_events`` (e.g. the
        worker was killed before the failure callback could append one), a synthetic
        terminal event is emitted so clients always observe an explicit ending.
        """

        run = self._store.get_run(run_id)
        if not run:
            raise BaselineError(ErrorCode.RUN_NOT_FOUND, "run not found", details={"run_id": run_id})
        self._assert_session_accessible(str(run.get("session_id") or ""))
        last_seq = max(0, int(after_seq))
        idle_deadline = time.time() + timeout_s
        next_keepalive = time.time() + keepalive_s
        while time.time() < idle_deadline:
            events = self._store.list_events(run_id, after_seq=last_seq)
            if events:
                now = time.time()
                idle_deadline = now + timeout_s
                next_keepalive = now + keepalive_s
            for event in events:
                last_seq = event["seq"]
                yield _sse(event["type"], event["payload"], event_id=last_seq)
                if event["type"] in {"workflow_end", "error"}:
                    return
            if not events:
                status = str((self._store.get_run(run_id) or {}).get("status") or "")
                if status in {
                    "completed",
                    "completed_degraded",
                    "cancelled",
                    "error",
                    "partial_timeout",
                    "paused",
                }:
                    # The terminal event may land just after our list_events read
                    # (the orchestrator updates the row before appending the event):
                    # give it one poll cycle, drain, then synthesize if still absent.
                    time.sleep(poll_s)
                    for event in self._store.list_events(run_id, after_seq=last_seq):
                        last_seq = event["seq"]
                        yield _sse(event["type"], event["payload"], event_id=last_seq)
                        if event["type"] in {"workflow_end", "error"}:
                            return
                    yield self._synthetic_terminal_event(run_id, status)
                    return
                if time.time() >= next_keepalive:
                    # SSE comment line: ignored by EventSource/fetch parsers but
                    # keeps intermediaries (nginx, browsers) from idling out.
                    yield b": keepalive\n\n"
                    next_keepalive = time.time() + keepalive_s
            time.sleep(poll_s)

    def _synthetic_terminal_event(self, run_id: str, status: str) -> bytes:
        """Build a terminal SSE frame for a run that ended without a terminal event."""

        if status in {"completed", "completed_degraded", "paused"}:
            return _sse("workflow_end", {"status": status})
        if status == "partial_timeout":
            return _sse("workflow_end", {"status": "partial_timeout", "code": ErrorCode.RUN_TIMEOUT.value})
        run = self._store.get_run(run_id) or {}
        payload = run.get("error")
        if not isinstance(payload, dict) or not payload:
            code = ErrorCode.RUN_CANCELLED if status == "cancelled" else ErrorCode.INTERNAL_ERROR
            payload = BaselineError(code, "run ended unexpectedly").to_payload()
        return _sse("error", payload)

    # --- sessions / memory / providers ---------------------------------------

    def list_sessions(self) -> list[JsonDict]:
        """List sessions, each annotated with its latest run id (for History)."""

        sessions = self._store.list_sessions()
        for session in sessions:
            latest = self._store.get_latest_run_for_session(session["id"])
            if latest:
                session["latest_run_id"], session["latest_run_mode"] = latest[0], latest[1]
                latest_run = self._store.get_run(latest[0]) or {}
                session["latest_run_status"] = latest_run.get("status")
                session["resumable_checkpoint"] = None
                if latest_run.get("status") == "paused":
                    checkpoint = self._store.get_deep_checkpoint(
                        latest[0],
                        include_payload=True,
                    )
                    if checkpoint is not None:
                        session["resumable_checkpoint"] = self._checkpoint_resume_view(
                            latest_run,
                            checkpoint,
                        )
            else:
                session["latest_run_id"] = None
                session["latest_run_mode"] = None
                session["latest_run_status"] = None
                session["resumable_checkpoint"] = None
        return [_public_session(session) for session in sessions]

    def list_messages(self, session_id: str) -> JsonDict:
        """List a session's persisted messages in order."""

        if not self._store.get_session(session_id):
            raise BaselineError(
                ErrorCode.SESSION_NOT_FOUND, "session not found", details={"session_id": session_id}
            )
        self._assert_session_accessible(session_id)
        return {"messages": self._store.list_messages(session_id)}

    def list_archived_sessions(self) -> list[JsonDict]:
        """List archived sessions (metadata only, no latest_run_id)."""

        sessions = self._store.list_sessions(archived=True)
        rows: list[JsonDict] = []
        for s in sessions:
            latest = self._store.get_latest_run_for_session(s["id"])
            rows.append(
                {
                    **_public_session(s),
                    "latest_run_mode": latest[1] if latest else None,
                }
            )
        return rows

    def patch_session(self, session_id: str, payload: JsonDict | None) -> JsonDict:
        """Update session-local UX preferences."""

        updates = _validate_session_patch_payload(payload)
        session = self._store.get_session(session_id)
        if not session:
            raise BaselineError(
                ErrorCode.SESSION_NOT_FOUND, "session not found", details={"session_id": session_id}
            )
        if session.get("archived_at") is not None:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST, "archived sessions cannot be patched", details={"session_id": session_id}
            )
        updated = self._store.update_session(
            session_id,
            pinned=updates.get("pinned"),
            mode_default=updates.get("mode_default"),
            update_pinned="pinned" in updates,
            update_mode_default="mode_default" in updates,
            metadata_patch=(
                {"search_usage_policy": updates["search_usage_policy"].to_dict()}
                if isinstance(updates.get("search_usage_policy"), SearchUsagePolicy)
                else None
            ),
            metadata_remove=(
                ("search_usage_policy",)
                if "search_usage_policy" in updates and updates.get("search_usage_policy") is None
                else ()
            ),
        )
        if updated is None:
            raise BaselineError(
                ErrorCode.SESSION_NOT_FOUND, "session not found", details={"session_id": session_id}
            )
        latest = self._store.get_latest_run_for_session(session_id)
        updated["latest_run_id"], updated["latest_run_mode"] = latest if latest else (None, None)
        return {"session_id": session_id, "session": _public_session(updated)}

    def archive_session(self, session_id: str) -> JsonDict:
        """Archive a session so it is hidden from active History."""

        if not self._store.get_session(session_id):
            raise BaselineError(
                ErrorCode.SESSION_NOT_FOUND, "session not found", details={"session_id": session_id}
            )
        if not self._store.archive_session(session_id):
            raise BaselineError(
                ErrorCode.INVALID_REQUEST, "session is already archived", details={"session_id": session_id}
            )
        return {"session_id": session_id, "archived": True}

    def restore_session(self, session_id: str) -> JsonDict:
        """Restore an archived session to active History."""

        if not self._store.get_session(session_id):
            raise BaselineError(
                ErrorCode.SESSION_NOT_FOUND, "session not found", details={"session_id": session_id}
            )
        if not self._store.restore_session(session_id):
            raise BaselineError(
                ErrorCode.INVALID_REQUEST, "session is not archived", details={"session_id": session_id}
            )
        return {"session_id": session_id, "archived": False}

    def delete_session(self, session_id: str) -> JsonDict:
        """Permanently delete a session and all session-scoped data."""

        if not self._store.delete_session(session_id):
            raise BaselineError(
                ErrorCode.SESSION_NOT_FOUND, "session not found", details={"session_id": session_id}
            )
        return {"session_id": session_id, "deleted": True}

    def create_session(self, title: str = "") -> JsonDict:
        """Create a session."""

        return {"session_id": self._store.create_session(title=title)}

    def _session_mode_default(self, session_id: str) -> str | None:
        if not session_id:
            return None
        session = self._store.get_session(session_id)
        if not session or session.get("archived_at") is not None:
            return None
        mode = str(session.get("mode_default") or "").lower()
        return mode if mode in {item.value for item in Mode} else None

    def add_memory(self, payload: JsonDict) -> JsonDict:
        """Add a memory item to a layer."""

        layer = MemoryLayer(str(payload.get("layer") or "session"))
        scope_key = str(payload.get("scope_key") or "")
        content = str(payload.get("content") or "")
        if not scope_key or not content:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "scope_key and content are required")
        mem_id = self._memory.remember(layer, scope_key, content, salience=float(payload.get("salience", 1.0)))
        return {"memory_id": mem_id}

    def recall_memory(self, payload: JsonDict) -> list[JsonDict]:
        """Recall memories across layers (optional ``query`` for semantic ranking)."""

        return self._memory.recall(
            session_key=str(payload.get("session_key") or ""),
            workspace_key=str(payload.get("workspace_key") or "default"),
            user_key=str(payload.get("user_key") or "default"),
            limit=int(payload.get("limit", 12)),
            query=str(payload.get("query") or ""),
        )

    def list_managed_memories(self, payload: JsonDict | None = None) -> list[JsonDict]:
        """List workspace/user memories for UI lifecycle management."""

        payload = payload or {}
        return self._memory_lifecycle.list_managed(
            workspace_key=str(payload.get("workspace_key") or "default"),
            user_key=str(payload.get("user_key") or "default"),
            limit=int(payload.get("limit", 100)),
            include_archived=bool(payload.get("include_archived", False)),
            archived_only=bool(payload.get("archived_only", False)),
        )

    def list_session_memories(self, payload: JsonDict | None = None) -> JsonDict:
        """Return grouped Session memories without invoking recall/ranking."""

        payload = payload or {}
        workspace_key = str(payload.get("workspace_key") or "default").strip() or "default"
        session_id = str(payload.get("session_id") or "").strip() or None
        query = str(payload.get("query") or "").strip()
        if len(query) > 200:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "query must be 200 characters or fewer")
        try:
            limit = int(payload.get("limit", 50))
            offset = int(payload.get("offset", 0))
        except (TypeError, ValueError) as error:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "limit and offset must be integers") from error
        if not 1 <= limit <= 100 or not 0 <= offset <= 10_000:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "limit must be 1..100 and offset must be 0..10000",
            )
        result = self._store.list_session_memory_groups(
            workspace_key=workspace_key,
            session_id=session_id,
            query=query,
            include_archived_sessions=bool(payload.get("include_archived_sessions", False)),
            include_archived_memories=bool(payload.get("include_archived_memories", False)),
            limit=limit,
            offset=offset,
        )
        for group in result["sessions"]:
            group["memories"] = [
                {
                    **memory,
                    "layer": "session",
                    "effective_salience": float(memory.get("effective_salience") or memory.get("salience") or 0),
                    "pinned": False,
                    "expires_at": None,
                }
                for memory in group.get("memories", [])
            ]
        return result

    def list_archived_memories(self, payload: JsonDict | None = None) -> JsonDict:
        """Return archived workspace/user memories."""

        return {"memories": self.list_managed_memories({**(payload or {}), "archived_only": True})}

    def archive_memory(self, memory_id: str) -> JsonDict:
        """Archive one workspace/user memory and purge its Library index."""

        return self._memory_lifecycle.archive_memory(memory_id)

    def restore_memory(self, memory_id: str) -> JsonDict:
        """Restore one workspace/user memory."""

        return self._memory_lifecycle.restore_memory(memory_id)

    def delete_memory(self, memory_id: str) -> JsonDict:
        """Permanently delete one workspace/user memory."""

        return self._memory_lifecycle.delete_memory(memory_id)

    def patch_memory(self, memory_id: str, payload: JsonDict | None = None) -> JsonDict:
        """Patch one active workspace/user memory."""

        return self._memory_lifecycle.patch_memory(memory_id, payload or {})

    def batch_memory_expiry(self, payload: JsonDict | None = None) -> JsonDict:
        """Apply an expiry value to active, unpinned workspace/user memories."""

        return self._memory_lifecycle.batch_expiry(payload or {})

    def cleanup_expired_memories(self, payload: JsonDict | None = None) -> JsonDict:
        """Archive a bounded batch of expired, unpinned memories."""

        return self._memory_lifecycle.cleanup_expired(payload or {})

    def get_memory(self, memory_id: str) -> JsonDict:
        """Return one canonical Memory row for citation deep links."""

        row = self._store.get_memory(str(memory_id or "").strip())
        if not row:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "memory not found")
        namespace = str(row.get("namespace") or "")
        return {
            **row,
            "layer": namespace,
            "effective_salience": float(row.get("salience") or 0.0),
            "pinned": bool(row.get("pinned")),
            "expires_at": row.get("expires_at"),
        }

    def preview_memory_optimization(self, payload: JsonDict | None = None) -> JsonDict:
        """Preview duplicate memory groups."""

        payload = payload or {}
        return self._memory_optimizer.preview_duplicates(
            workspace_key=str(payload.get("workspace_key") or "default"),
            user_key=str(payload.get("user_key") or "default"),
            threshold=payload.get("threshold", 0.82),
            limit=int(payload.get("limit", 200)),
        )

    def apply_memory_optimization(self, payload: JsonDict | None = None) -> JsonDict:
        """Apply a confirmed memory optimization action."""

        return self._memory_optimizer.apply(payload or {})

    def get_memory_dream_status(self, payload: JsonDict | None = None) -> JsonDict:
        """Return the latest Dreaming v2 status for a workspace."""

        payload = payload or {}
        workspace_key = str(payload.get("workspace_key") or "default")
        user_key = str(payload.get("user_key") or "default")
        self._reconcile_memory_dream_runs(workspace_key=workspace_key)
        if bool(payload.get("auto_trigger", True)):
            self._maybe_start_memory_dream_schedule(workspace_key=workspace_key, user_key=user_key)
        row = self._store.get_latest_memory_dream_run(workspace_key=workspace_key)
        if not row:
            return self._with_dream_schedule(
                {
                "status": "idle",
                "dream_run_id": None,
                "workspace_key": workspace_key,
                "user_key": user_key,
                "started_at": None,
                "finished_at": None,
                "memories_created": 0,
                "error": None,
                },
                None,
            )
        return self._with_dream_schedule(_dream_status_payload(row), row)

    def get_memory_write_audit(self, payload: JsonDict | None = None) -> JsonDict:
        """Return the latest secret-free Memory write audit summary."""

        payload = payload or {}
        workspace_key = str(payload.get("workspace_key") or "default").strip() or "default"
        user_key = str(payload.get("user_key") or "default").strip() or "default"
        candidates: list[JsonDict] = []
        row = self._store.get_latest_memory_dream_run(workspace_key=workspace_key)
        if row:
            candidates.append(_audit_from_dream_status(_dream_status_payload(row)))
        turn_audit = self._latest_post_turn_audit(workspace_key=workspace_key)
        if turn_audit:
            candidates.append(turn_audit)
        if not candidates:
            probe = self._memory_settings.format_probe if isinstance(self._memory_settings.format_probe, dict) else {}
            return {
                "status": "idle",
                "source": "none",
                "workspace_key": workspace_key,
                "user_key": user_key,
                "attempted": 0,
                "succeeded": 0,
                "skipped": 0,
                "failed": 0,
                "error_code": None,
                "checked_at": None,
                "provider_ref": self._memory_provider_ref(),
                "stale": bool(probe.get("stale")),
                "incomplete_reasons": [],
            }
        candidates.sort(key=lambda item: float(item.get("checked_at") or 0), reverse=True)
        latest = candidates[0]
        return {
            "workspace_key": workspace_key,
            "user_key": user_key,
            **latest,
        }

    def _latest_post_turn_audit(self, *, workspace_key: str) -> JsonDict | None:
        persisted = self._store.get_latest_post_turn_memory_run(workspace_key=workspace_key)
        if persisted:
            status = str(persisted.get("status") or "failed")
            return {
                "status": status,
                "source": "post_turn_memory_v1",
                "attempted": 1,
                "succeeded": int(status == "completed"),
                "skipped": 0,
                "failed": int(status == "failed"),
                "error_code": persisted.get("error_code"),
                "checked_at": persisted.get("checked_at"),
                "provider_ref": str(persisted.get("provider_ref") or ""),
                "run_id": str(persisted.get("run_id") or ""),
                "session_id": str(persisted.get("session_id") or ""),
                "embedding_status": persisted.get("embedding_status"),
                "stale": False,
                "incomplete_reasons": [],
            }
        rows = self._store.list_memories(namespace=MemoryLayer.WORKSPACE.value, scope_key=workspace_key, limit=1000)
        latest: JsonDict | None = None
        for row in rows:
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            if metadata.get("source") != "post_turn_memory_v1":
                continue
            checked_at = float(metadata.get("checked_at") or row.get("last_used_at") or row.get("created_at") or 0)
            audit: JsonDict = {
                "status": "completed",
                "source": "post_turn_memory_v1",
                "attempted": 1,
                "succeeded": 1,
                "skipped": 0,
                "failed": 0,
                "error_code": None,
                "checked_at": checked_at,
                "provider_ref": self._memory_provider_ref(),
                "stale": False,
                "incomplete_reasons": [],
            }
            if latest is None or checked_at > float(latest.get("checked_at") or 0):
                latest = audit
        return latest

    def update_memory_dream_schedule(self, payload: JsonDict | None = None) -> JsonDict:
        """Persist the workspace Dreaming schedule."""

        payload = payload or {}
        try:
            schedule = normalize_memory_dream_schedule(payload.get("schedule"))
        except ValueError as error:
            raise BaselineError(ErrorCode.INVALID_REQUEST, str(error)) from error
        self._memory_dream_schedule = schedule
        if self._config_path is not None:
            save_memory_dream_schedule_config(self._config_path, schedule)
        status = self.get_memory_dream_status({"auto_trigger": False})
        return {
            "schedule": schedule,
            "next_scheduled_at": status.get("next_scheduled_at"),
        }

    def start_memory_dream(self, payload: JsonDict | None = None, *, background: bool | None = None) -> JsonDict:
        """Start memory continuity distillation and return a safe status payload."""

        payload = payload or {}
        if not self._memory_distiller.is_configured():
            raise BaselineError(
                ErrorCode.PROVIDER_NOT_CONFIGURED,
                "memory processing model is not configured",
                details={"role": "memory"},
            )
        workspace_key = str(payload.get("workspace_key") or "default").strip() or "default"
        user_key = str(payload.get("user_key") or "default").strip() or "default"
        session_id = str(payload.get("session_id") or "").strip() or None
        run_id = str(payload.get("run_id") or "").strip()
        include_facts = bool(payload.get("include_facts", False))
        strategy = self._memory_settings.workspace_memory_refresh_strategy
        provider_ref = self._memory_provider_ref()
        thinking_budget = self._resolve_memory_thinking_budget()
        run_background = bool(payload.get("background", True)) if background is None else background
        backend = self._job_backend.name if run_background else "sync"
        self._reconcile_memory_dream_runs(workspace_key=workspace_key)
        metadata: JsonDict = {
            "source": "memory_continuity_v1",
            "session_id": session_id,
            "run_id": run_id,
            "include_facts": include_facts,
            "workspace_memory_refresh_strategy": strategy,
            "provider_ref": provider_ref,
            "thinking_budget": thinking_budget,
        }
        reservation = self._store.create_memory_dream_run_if_idle(
            workspace_key=workspace_key,
            user_key=user_key,
            status="queued" if run_background else "running",
            backend=backend,
            metadata=metadata,
        )
        row = reservation.get("run") if isinstance(reservation.get("run"), dict) else {}
        if not reservation.get("created"):
            raise BaselineError(
                ErrorCode.MEMORY_DREAM_RUNNING,
                "a Memory Dream is already active for this workspace",
                details={
                    "workspace_key": workspace_key,
                    "dream_run_id": row.get("id"),
                    "status": row.get("status"),
                },
            )
        dream_id = str(row.get("id") or "")
        job_payload = {
            "workspace_key": workspace_key,
            "user_key": user_key,
            "session_id": session_id,
            "run_id": run_id,
            "include_facts": include_facts,
            "strategy": strategy,
            "provider_ref": provider_ref,
            "thinking_budget": thinking_budget,
        }
        if run_background:
            try:
                backend_job_id = self._job_backend.enqueue_memory_dream(job_payload, dream_id)
                self._store.set_memory_dream_backend_job_id(dream_id, backend_job_id)
            except Exception as error:  # noqa: BLE001 - enqueue failure must become terminal
                self._store.update_memory_dream_run(
                    dream_id,
                    status="error",
                    expected_statuses={"queued"},
                    error="Memory Dream could not be queued.",
                    metadata={"error_code": "MEMORY_DREAM_ENQUEUE_FAILED", "incomplete": True},
                )
                raise BaselineError(
                    ErrorCode.INTERNAL_ERROR,
                    "Memory Dream could not be queued",
                    details={"dream_run_id": dream_id, "workspace_key": workspace_key},
                ) from error
            row = self._store.get_memory_dream_run(dream_id) or row
            return self._with_dream_schedule(_dream_status_payload(row), row)
        self._run_memory_dream_safe(
            dream_id,
            workspace_key,
            user_key,
            session_id,
            strategy,
            provider_ref,
            include_facts,
            run_id,
        )
        row = self._store.get_memory_dream_run(dream_id) or {}
        return self._with_dream_schedule(_dream_status_payload(row), row)

    def execute_memory_dream_job(self, payload: JsonDict, dream_run_id: str) -> None:
        """Execute a persisted background Dream from thread or RQ."""

        row = self._store.get_memory_dream_run(dream_run_id)
        if not row or str(row.get("status") or "") in _DREAM_TERMINAL_STATUSES:
            return
        if str(row.get("status") or "") == "cancelling":
            self._store.update_memory_dream_run(
                dream_run_id,
                status="cancelled",
                expected_statuses={"cancelling"},
                metadata={"error_code": "MEMORY_DREAM_CANCELLED"},
            )
            return
        if not self._store.update_memory_dream_run(
            dream_run_id,
            status="running",
            expected_statuses={"queued"},
            metadata={"stage": "distilling"},
        ):
            return

        provider_ref = str(payload.get("provider_ref") or "").strip()
        thinking_budget = payload.get("thinking_budget")
        distiller = self._memory_distiller
        if provider_ref:
            adapter = self._adapter_for_ref(provider_ref)
            if adapter is None:
                self.fail_memory_dream(
                    dream_run_id,
                    "The selected Memory processing model is no longer available.",
                    error_code="MEMORY_PROVIDER_UNAVAILABLE",
                )
                return
            distiller = MemoryDistillationService(
                store=self._store,
                memory_llm=lambda: adapter,
                memory_thinking_budget=lambda: thinking_budget if isinstance(thinking_budget, int) else None,
            )
        self._run_memory_dream_safe(
            dream_run_id,
            str(payload.get("workspace_key") or "default"),
            str(payload.get("user_key") or "default"),
            str(payload.get("session_id") or "").strip() or None,
            str(payload.get("strategy") or "incremental"),
            provider_ref,
            bool(payload.get("include_facts", False)),
            str(payload.get("run_id") or ""),
            distiller=distiller,
        )

    def cancel_memory_dream(self, dream_run_id: str) -> JsonDict:
        """Request best-effort cancellation of one queued/running Dream."""

        row = self._store.get_memory_dream_run(str(dream_run_id or "").strip())
        if row is None:
            raise BaselineError(
                ErrorCode.MEMORY_DREAM_NOT_FOUND,
                "Memory Dream not found",
                details={"dream_run_id": str(dream_run_id or "")},
            )
        status = str(row.get("status") or "")
        if status in _DREAM_TERMINAL_STATUSES:
            return self._with_dream_schedule(_dream_status_payload(row), row)
        if str(row.get("backend") or "") == "sync":
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "synchronous Memory Dreams cannot be cancelled asynchronously",
                details={"dream_run_id": dream_run_id},
            )
        self._store.update_memory_dream_run(
            dream_run_id,
            status="cancelling",
            expected_statuses=_DREAM_ACTIVE_STATUSES,
            memories_created=int(row.get("memories_created") or 0),
            cancel_requested=True,
            metadata={"stage": "cancelling"},
        )
        backend_job_id = str(row.get("backend_job_id") or "")
        outcome = "cancelling"
        if backend_job_id:
            try:
                outcome = self._job_backend.cancel_memory_dream(backend_job_id)
            except Exception:  # noqa: BLE001 - durable cancel flag remains authoritative
                outcome = "cancelling"
        if outcome in {"cancelled", "canceled"}:
            self._store.update_memory_dream_run(
                dream_run_id,
                status="cancelled",
                expected_statuses={"cancelling"},
                memories_created=int(row.get("memories_created") or 0),
                metadata={"stage": "cancelled", "error_code": "MEMORY_DREAM_CANCELLED"},
            )
        updated = self._store.get_memory_dream_run(dream_run_id) or row
        return self._with_dream_schedule(_dream_status_payload(updated), updated)

    def fail_memory_dream(
        self,
        dream_run_id: str,
        message: str,
        *,
        error_code: str = "MEMORY_DREAM_WORKER_FAILED",
    ) -> None:
        """Close an active Dream after an observable worker failure."""

        row = self._store.get_memory_dream_run(dream_run_id)
        if not row:
            return
        current = str(row.get("status") or "")
        if current == "cancelling":
            self._store.update_memory_dream_run(
                dream_run_id,
                status="cancelled",
                expected_statuses={"cancelling"},
                memories_created=int(row.get("memories_created") or 0),
                metadata={"error_code": "MEMORY_DREAM_CANCELLED", "stage": "cancelled"},
            )
            return
        self._store.update_memory_dream_run(
            dream_run_id,
            status="error",
            expected_statuses={"queued", "running"},
            memories_created=int(row.get("memories_created") or 0),
            error=_safe_status_error(message),
            metadata={"error_code": error_code, "incomplete": True, "stage": "failed"},
        )

    def _reconcile_memory_dream_runs(self, *, workspace_key: str | None = None) -> None:
        """Close observable orphan jobs while preserving live queued/running work."""

        now = time.time()
        for row in self._store.list_active_memory_dream_runs(workspace_key=workspace_key):
            dream_run_id = str(row.get("id") or "")
            current = str(row.get("status") or "")
            backend = str(row.get("backend") or "thread")
            backend_job_id = str(row.get("backend_job_id") or "")
            updated_at = float(row.get("updated_at") or row.get("started_at") or now)
            stale_enough = now - updated_at >= _DREAM_RECONCILE_GRACE_S
            if backend == "sync":
                continue
            if not backend_job_id:
                if stale_enough:
                    self.fail_memory_dream(
                        dream_run_id,
                        "Memory Dream lost its backend job before execution.",
                        error_code="MEMORY_DREAM_JOB_MISSING",
                    )
                continue
            try:
                backend_status = self._job_backend.memory_dream_job_status(backend_job_id)
            except Exception:  # noqa: BLE001 - backend outage is not proof the job died
                backend_status = "unknown"
            if backend_status in {"queued", "deferred", "scheduled", "started", "running", "unknown"}:
                continue
            if backend_status in {"canceled", "cancelled"}:
                self._store.update_memory_dream_run(
                    dream_run_id,
                    status="cancelled",
                    expected_statuses=_DREAM_ACTIVE_STATUSES,
                    memories_created=int(row.get("memories_created") or 0),
                    metadata={"stage": "cancelled", "error_code": "MEMORY_DREAM_CANCELLED"},
                )
                continue
            if backend_status == "stopped" and current == "cancelling":
                self._store.update_memory_dream_run(
                    dream_run_id,
                    status="cancelled",
                    expected_statuses={"cancelling"},
                    memories_created=int(row.get("memories_created") or 0),
                    metadata={"stage": "cancelled", "error_code": "MEMORY_DREAM_CANCELLED"},
                )
                continue
            if backend_status in {"failed", "stopped"}:
                self.fail_memory_dream(
                    dream_run_id,
                    "Memory Dream backend reported a terminal failure.",
                    error_code="MEMORY_DREAM_BACKEND_FAILED",
                )
                continue
            if backend_status in {None, "finished"} and stale_enough:
                self.fail_memory_dream(
                    dream_run_id,
                    "Memory Dream backend state was lost before a terminal status was persisted.",
                    error_code="MEMORY_DREAM_ORPHANED",
                )

    def refresh_post_turn_memory(
        self,
        *,
        session_id: str,
        run_id: str = "",
        workspace_key: str = "default",
        user_key: str = "default",
    ) -> JsonDict:
        """Capture one completed turn as Memory without running full session distillation."""

        result = self._memory_distiller.refresh_post_turn(
            workspace_key=workspace_key,
            user_key=user_key,
            session_id=session_id,
            run_id=run_id,
        )
        result["provider_ref"] = self._memory_provider_ref()
        return result

    def _run_memory_dream_safe(
        self,
        dream_run_id: str,
        workspace_key: str,
        user_key: str,
        session_id: str | None = None,
        strategy: str | None = None,
        provider_ref: str = "",
        include_facts: bool = False,
        run_id: str = "",
        *,
        distiller: MemoryDistillationService | None = None,
    ) -> None:
        """Execute memory continuity distillation and persist a terminal status."""

        selected_strategy = strategy or self._memory_settings.workspace_memory_refresh_strategy
        safe_provider_ref = provider_ref or self._memory_provider_ref()
        active_distiller = distiller or self._memory_distiller
        should_cancel = lambda: self._memory_dream_cancel_requested(dream_run_id)
        try:
            if should_cancel():
                raise MemoryDistillationCancelled("Memory Dream cancellation requested")
            self._store.sweep_expired_memories(
                namespace=MemoryLayer.WORKSPACE.value,
                scope_key=workspace_key,
                limit=500,
            )
            self._store.sweep_expired_memories(
                namespace=MemoryLayer.USER.value,
                scope_key=user_key,
                limit=500,
            )
            result = active_distiller.refresh(
                workspace_key=workspace_key,
                user_key=user_key,
                session_id=session_id,
                include_user=True,
                strategy=selected_strategy,
                should_cancel=should_cancel,
            )
            if include_facts:
                fact_result = active_distiller.refresh_memory_facts(
                    workspace_key=workspace_key,
                    user_key=user_key,
                    session_id=session_id,
                    run_id=run_id,
                    should_cancel=should_cancel,
                )
                result = {
                    **result,
                    "memories_created": int(result.get("memories_created") or 0)
                    + int(fact_result.get("memories_created") or 0),
                    "accepted": int(fact_result.get("accepted") or 0),
                    "skipped": int(fact_result.get("skipped") or 0),
                    "failed": int(fact_result.get("failed") or 0),
                    "fact_attempted": int(fact_result.get("attempted") or 0),
                    "fact_source": "memory_fact_v1",
                }
            if should_cancel():
                raise MemoryDistillationCancelled("Memory Dream cancellation requested")
            self._store.update_memory_dream_run(
                dream_run_id,
                status="completed",
                expected_statuses={"running"},
                memories_created=int(result.get("memories_created") or 0),
                metadata={
                    "source": "memory_continuity_v1",
                    **result,
                    "session_id": session_id,
                    "run_id": run_id,
                    "include_facts": include_facts,
                    "workspace_memory_refresh_strategy": selected_strategy,
                    "provider_ref": safe_provider_ref,
                },
            )
        except MemoryDistillationCancelled:
            current = self._store.get_memory_dream_run(dream_run_id) or {}
            self._store.update_memory_dream_run(
                dream_run_id,
                status="cancelled",
                expected_statuses={"running", "cancelling"},
                memories_created=int(current.get("memories_created") or 0),
                metadata={
                    "source": "memory_continuity_v1",
                    "stage": "cancelled",
                    "error_code": "MEMORY_DREAM_CANCELLED",
                    "session_id": session_id,
                    "run_id": run_id,
                    "provider_ref": safe_provider_ref,
                },
            )
        except Exception as error:  # noqa: BLE001 - Dreaming is isolated from research runs
            if should_cancel():
                current = self._store.get_memory_dream_run(dream_run_id) or {}
                self._store.update_memory_dream_run(
                    dream_run_id,
                    status="cancelled",
                    expected_statuses={"running", "cancelling"},
                    memories_created=int(current.get("memories_created") or 0),
                    metadata={"stage": "cancelled", "error_code": "MEMORY_DREAM_CANCELLED"},
                )
                return
            metadata: JsonDict = {
                "source": "memory_continuity_v1",
                "session_id": session_id,
                "run_id": run_id,
                "include_facts": include_facts,
                "workspace_memory_refresh_strategy": selected_strategy,
                "provider_ref": safe_provider_ref,
            }
            if isinstance(error, MemoryDistillationError):
                metadata.update(error.status_metadata())
            else:
                metadata.update({"error_code": "MEMORY_DISTILLATION_FAILED", "failed_stage": "unknown"})
            metadata["incomplete"] = True
            metadata["incomplete_reasons"] = [str(metadata.get("error_code") or "MEMORY_DISTILLATION_FAILED")]
            self._store.update_memory_dream_run(
                dream_run_id,
                status="error",
                expected_statuses={"running"},
                memories_created=0,
                error=_safe_status_error(str(error) or error.__class__.__name__),
                metadata=metadata,
            )

    def _memory_dream_cancel_requested(self, dream_run_id: str) -> bool:
        row = self._store.get_memory_dream_run(dream_run_id) or {}
        return str(row.get("status") or "") == "cancelling" or row.get("cancel_requested_at") is not None

    def _with_dream_schedule(self, status: JsonDict, row: JsonDict | None) -> JsonDict:
        """Attach schedule metadata to a Dreaming status payload."""

        next_at = _dream_next_scheduled_at(row, self._memory_dream_schedule)
        return {
            **status,
            "schedule": self._memory_dream_schedule,
            "next_scheduled_at": next_at,
            "schedule_due": bool(next_at is not None and time.time() >= next_at),
        }

    def _maybe_start_memory_dream_schedule(
        self,
        *,
        workspace_key: str,
        user_key: str,
        background: bool = True,
    ) -> None:
        """Best-effort lazy scheduled Dreaming trigger."""

        if self._memory_dream_schedule == "off" or not self._memory_distiller.is_configured():
            return
        row = self._store.get_latest_memory_dream_run(workspace_key=workspace_key)
        next_at = _dream_next_scheduled_at(row, self._memory_dream_schedule)
        if next_at is None or time.time() < next_at:
            return
        try:
            self.start_memory_dream(
                {"workspace_key": workspace_key, "user_key": user_key, "background": background},
                background=background,
            )
        except Exception:  # noqa: BLE001 - scheduled Dreaming never blocks status reads
            return

    def search_history(self, session_id: str, payload: JsonDict) -> JsonDict:
        """Semantic search over conversation history (intra/cross conversation)."""

        if not self._store.get_session(session_id):
            raise BaselineError(
                ErrorCode.SESSION_NOT_FOUND, "session not found", details={"session_id": session_id}
            )
        self._assert_session_accessible(session_id)
        query = str(payload.get("query") or "").strip()
        if not query:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "query is required")
        scope = str(payload.get("scope") or SCOPE_CROSS)
        if scope not in (SCOPE_INTRA, SCOPE_CROSS):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "scope must be 'intra' or 'cross'")
        limit = int(payload.get("limit", 10))
        results = self._history.search_history(query, session_id=session_id, scope=scope, limit=limit)
        return {"results": results, "scope": scope}

    def provider_status(self) -> JsonDict:
        """Return secret-free provider status."""

        with self._provider_config_lock:
            return self._provider_status()

    def update_providers(self, payload: JsonDict) -> JsonDict:
        """Apply BYOK provider updates at runtime, persist them, and return secret-free status.

        Non-empty values from ``payload`` (restricted to known provider fields) are merged
        onto the current settings, the runtime is rebuilt, and the result is persisted to the
        local config file when one is configured. Secrets are never logged or returned.
        """

        if not isinstance(payload, dict):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "provider payload must be an object")
        search_updates = {field: payload[field] for field in _SEARCH_PROVIDER_FIELDS if field in payload}
        other_updates = {
            field: payload[field]
            for field in PROVIDER_FIELDS
            if field in payload and field not in _SEARCH_PROVIDER_FIELDS
        }
        if search_updates and other_updates:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "Search provider settings must be saved separately from LLM settings.",
            )
        if search_updates:
            providers: JsonDict = {}
            for provider, field in _SEARCH_PROVIDER_KEY_FIELDS.items():
                entry: JsonDict = {}
                if field in search_updates:
                    entry["api_key"] = search_updates[field]
                if provider == "firecrawl" and "firecrawl_base_url" in search_updates:
                    entry["base_url"] = search_updates["firecrawl_base_url"]
                if entry:
                    providers[provider] = entry
            return self.save_search_providers(
                {
                    "expected_revision": int(self._search_provider_state.get("revision") or 0),
                    "providers": providers,
                }
            )
        updates = other_updates
        with self._provider_config_lock:
            provider = merge_provider_config(self._provider, updates)
            # Persist before swapping the live runtime. A disk/config failure must
            # leave the active provider set unchanged, and the service lock prevents
            # concurrent partial PATCH requests from losing each other's fields.
            if self._config_path is not None:
                save_non_search_provider_config(self._config_path, provider)
            self.reconfigure(provider)
            return self._provider_status()

    def save_search_providers(self, payload: JsonDict, *, opener=None) -> JsonDict:
        """Probe and partially persist live External-search provider settings."""

        if not isinstance(payload, dict) or not isinstance(payload.get("providers"), dict):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "providers must be an object")
        if isinstance(payload.get("expected_revision"), bool):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "expected_revision is required")
        try:
            expected_revision = int(payload.get("expected_revision"))
        except (TypeError, ValueError) as error:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "expected_revision is required") from error
        if expected_revision < 0:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "expected_revision must be non-negative")
        raw_entries = payload["providers"]
        unknown = sorted(set(raw_entries) - set(SEARCH_PROVIDER_IDS))
        if unknown:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                f"unknown search provider: {', '.join(unknown)}",
            )

        with self._provider_config_lock:
            actual_revision = int(self._search_provider_state.get("revision") or 0)
            if expected_revision != actual_revision:
                raise self._search_provider_conflict(expected_revision, actual_revision)
            current_provider = self._provider
            current_sources = dict(self._search_provider_sources)
            current_state = normalize_state_block(self._search_provider_state)

        entries: dict[str, JsonDict] = {}
        changed: dict[str, bool] = {}
        candidate_updates: dict[str, str] = {}
        removals: set[str] = set()
        for provider in SEARCH_PROVIDER_IDS:
            raw = raw_entries.get(provider, {})
            if not isinstance(raw, dict):
                raise BaselineError(
                    ErrorCode.INVALID_REQUEST,
                    f"{provider} settings must be an object",
                )
            allowed = {"api_key", "remove"} | ({"base_url"} if provider == "firecrawl" else set())
            unsupported = sorted(set(raw) - allowed)
            if unsupported:
                raise BaselineError(
                    ErrorCode.INVALID_REQUEST,
                    f"unsupported {provider} fields: {', '.join(unsupported)}",
                )
            if "remove" in raw and not isinstance(raw["remove"], bool):
                raise BaselineError(
                    ErrorCode.INVALID_REQUEST,
                    f"{provider} remove must be a boolean",
                )
            for text_field in ("api_key", "base_url"):
                if text_field in raw and raw[text_field] is not None and not isinstance(raw[text_field], str):
                    raise BaselineError(
                        ErrorCode.INVALID_REQUEST,
                        f"{provider} {text_field} must be a string",
                    )
            remove = raw.get("remove", False)
            api_key = str(raw.get("api_key") or "").strip()
            base_url = str(raw.get("base_url") or "").strip()
            if remove and api_key:
                raise BaselineError(
                    ErrorCode.INVALID_REQUEST,
                    f"{provider} remove and api_key are mutually exclusive",
                )
            source = current_sources.get(provider, "none")
            if (remove or api_key) and source in {"env", "secure"}:
                raise BaselineError(
                    ErrorCode.CREDENTIAL_SOURCE_READ_ONLY,
                    f"{provider} credentials are managed by a read-only source",
                    details={"provider": provider, "credential_source": source},
                )
            field = _SEARCH_PROVIDER_KEY_FIELDS[provider]
            previous_key = str(getattr(current_provider, field) or "").strip()
            provider_changed = remove or bool(api_key and api_key != previous_key)
            if provider == "firecrawl" and base_url:
                provider_changed = provider_changed or base_url != str(current_provider.firecrawl_base_url or "").strip()
            entries[provider] = {
                "api_key": api_key,
                "base_url": base_url,
                "remove": remove,
            }
            changed[provider] = provider_changed
            if remove:
                removals.add(provider)
            elif api_key:
                candidate_updates[field] = api_key
            if provider == "firecrawl" and base_url:
                candidate_updates["firecrawl_base_url"] = base_url

        candidate = replace(current_provider, **candidate_updates)
        for provider in removals:
            candidate = replace(candidate, **{_SEARCH_PROVIDER_KEY_FIELDS[provider]: ""})

        state_items = current_state["providers"]
        to_probe: list[str] = []
        for provider in SEARCH_PROVIDER_IDS:
            enabled = bool(state_items[provider].get("enabled", True))
            field = _SEARCH_PROVIDER_KEY_FIELDS[provider]
            if provider in removals or not _configured_value(str(getattr(candidate, field) or "")):
                continue
            if enabled or changed[provider]:
                to_probe.append(provider)
        probes = self._probe_search_provider_batch(candidate, to_probe, opener=opener)

        provider_updates: dict[str, str | None] = {}
        next_provider = current_provider
        next_states = normalize_state_block(current_state)["providers"]
        volatile_updates: dict[str, JsonDict] = {}
        saved: list[str] = []
        rejected: list[str] = []
        response_meta: dict[str, JsonDict] = {}
        persistent_change = False
        for provider in SEARCH_PROVIDER_IDS:
            entry = entries[provider]
            source = current_sources.get(provider, "none")
            old_item = dict(next_states[provider])
            old_revision = int(old_item.get("credential_revision") or 0)
            probe = probes.get(provider)
            draft_changed = changed[provider]
            accepted = False
            stored_retained = bool(
                _configured_value(str(getattr(current_provider, _SEARCH_PROVIDER_KEY_FIELDS[provider]) or ""))
            )
            probe_persisted = False

            if provider in removals:
                new_revision = old_revision + 1
                provider_updates[_SEARCH_PROVIDER_KEY_FIELDS[provider]] = None
                next_provider = replace(next_provider, **{_SEARCH_PROVIDER_KEY_FIELDS[provider]: ""})
                next_states[provider] = {
                    "enabled": bool(old_item.get("enabled", True)),
                    "credential_revision": new_revision,
                }
                saved.append(provider)
                current_sources[provider] = "none"
                accepted = True
                stored_retained = False
                persistent_change = True
            elif draft_changed and (probe is None or not probe_can_persist_draft(probe)):
                rejected.append(provider)
            elif draft_changed and source in {"env", "secure"}:
                # Only Firecrawl base_url can reach this branch; credential
                # drafts/removal were rejected above. Persist the endpoint while
                # keeping the overlay probe process-local.
                base_url = str(entry.get("base_url") or "").strip()
                if base_url:
                    provider_updates["firecrawl_base_url"] = base_url
                    next_provider = replace(next_provider, firecrawl_base_url=base_url)
                volatile_updates[provider] = probe_snapshot(
                    probe or {}, credential_revision=None
                )
                saved.append(provider)
                accepted = True
                persistent_change = True
            elif draft_changed:
                new_revision = old_revision + 1
                field = _SEARCH_PROVIDER_KEY_FIELDS[provider]
                key = str(entry.get("api_key") or "").strip()
                if key:
                    provider_updates[field] = key
                    next_provider = replace(next_provider, **{field: key})
                if provider == "firecrawl" and str(entry.get("base_url") or "").strip():
                    base_url = str(entry["base_url"]).strip()
                    provider_updates["firecrawl_base_url"] = base_url
                    next_provider = replace(next_provider, firecrawl_base_url=base_url)
                item = {
                    "enabled": bool(old_item.get("enabled", True)),
                    "credential_revision": new_revision,
                    "last_probe": probe_snapshot(
                        probe or {}, credential_revision=new_revision
                    ),
                }
                next_states[provider] = item
                saved.append(provider)
                if source not in {"env", "secure"}:
                    current_sources[provider] = "config"
                accepted = True
                stored_retained = False
                probe_persisted = source not in {"env", "secure"}
                persistent_change = True
            elif probe is not None and source in {"env", "secure"}:
                volatile_updates[provider] = probe_snapshot(
                    probe, credential_revision=None
                )
            elif probe is not None:
                # Existing config credentials are re-tested on every Save. A
                # legacy credential receives its first revision even on a soft
                # failure so future run snapshots have a stable identity.
                new_revision = old_revision or 1
                next_states[provider] = {
                    "enabled": bool(old_item.get("enabled", True)),
                    "credential_revision": new_revision,
                    "last_probe": probe_snapshot(
                        probe, credential_revision=new_revision
                    ),
                }
                probe_persisted = True
                persistent_change = True

            response_meta[provider] = {
                "draft_accepted": accepted,
                "stored_key_retained": stored_retained and not accepted,
                "probe_persisted": probe_persisted,
            }

        try:
            if persistent_change and self._config_path is not None:
                committed = commit_search_provider_config(
                    self._config_path,
                    expected_revision=expected_revision,
                    provider_updates=provider_updates,
                    provider_states=next_states,
                )
            elif persistent_change:
                committed = normalize_state_block(
                    {
                        "revision": expected_revision + 1,
                        "providers": next_states,
                    }
                )
            else:
                if self._config_path is not None:
                    latest = load_search_provider_state_config(self._config_path)
                    actual = int(latest.get("revision") or 0)
                    if actual != expected_revision:
                        raise SearchProviderStateConflict(expected=expected_revision, actual=actual)
                committed = current_state
        except SearchProviderStateConflict as error:
            raise self._search_provider_conflict(error.expected, error.actual) from error

        runtime_search_updates = {
            field: getattr(next_provider, field)
            for field in _SEARCH_PROVIDER_FIELDS
            if getattr(next_provider, field) != getattr(current_provider, field)
        }
        with self._provider_config_lock:
            self._search_provider_state = committed
            self._volatile_search_probes.update(volatile_updates)
            runtime_provider = replace(self._provider, **runtime_search_updates)
            if runtime_provider != self._provider:
                # A non-Search settings write may have completed while network
                # probes were in flight. Apply only this Save's accepted Search
                # fields so that runtime state cannot roll that concurrent write
                # back after the full-file CAS commit preserved it on disk.
                self._provider = runtime_provider
                self._search_provider_sources = self._normalize_search_provider_sources(
                    current_sources
                )
                self._wire_runtime()
            states = self._search_provider_states()
            public_status = self._provider_status()
        for provider in SEARCH_PROVIDER_IDS:
            states[provider].update(response_meta[provider])

        if saved and rejected:
            operation_status = "partial"
        elif rejected:
            operation_status = "rejected"
        elif saved or probes:
            operation_status = "success"
        else:
            operation_status = "noop"
        return {
            "operation_status": operation_status,
            "revision": int(committed.get("revision") or 0),
            "providers": states,
            "saved": saved,
            "rejected": rejected,
            "external_search_service": public_status["external_search_service"],
            "banner": public_status["external_search_service"]["banner"],
        }

    def toggle_search_provider(self, payload: JsonDict) -> JsonDict:
        """Persist one enabled flag with CAS; only subsequently created runs observe it."""

        if not isinstance(payload, dict):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "toggle payload must be an object")
        provider = str(payload.get("provider") or "").strip().lower()
        if (
            provider not in SEARCH_PROVIDER_IDS
            or "enabled" not in payload
            or not isinstance(payload.get("enabled"), bool)
        ):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "provider and enabled are required")
        if isinstance(payload.get("expected_revision"), bool):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "expected_revision is required")
        try:
            expected_revision = int(payload.get("expected_revision"))
        except (TypeError, ValueError) as error:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "expected_revision is required") from error
        if expected_revision < 0:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "expected_revision must be non-negative")
        enabled = payload["enabled"]
        with self._provider_config_lock:
            current = normalize_state_block(self._search_provider_state)
            actual = int(current.get("revision") or 0)
            if actual != expected_revision:
                raise self._search_provider_conflict(expected_revision, actual)
            states = current["providers"]
            if bool(states[provider].get("enabled", True)) == enabled:
                return {
                    "operation_status": "noop",
                    "revision": actual,
                    "providers": self._provider_status()["search_providers"],
                    "external_search_service": self._provider_status()["external_search_service"],
                }
            states[provider] = {**states[provider], "enabled": enabled}
        try:
            if self._config_path is not None:
                committed = commit_search_provider_config(
                    self._config_path,
                    expected_revision=expected_revision,
                    provider_updates={},
                    provider_states=states,
                )
            else:
                committed = normalize_state_block(
                    {"revision": expected_revision + 1, "providers": states}
                )
        except SearchProviderStateConflict as error:
            raise self._search_provider_conflict(error.expected, error.actual) from error
        with self._provider_config_lock:
            self._search_provider_state = committed
            status = self._provider_status()
        return {
            "operation_status": "success",
            "revision": int(committed.get("revision") or 0),
            "providers": status["search_providers"],
            "external_search_service": status["external_search_service"],
        }

    def _probe_search_provider_batch(
        self,
        candidate: ProviderSettings,
        providers: list[str],
        *,
        opener=None,
    ) -> dict[str, JsonDict]:
        """Probe at most four providers without holding config or runtime locks."""

        if not providers:
            return {}
        executor = ThreadPoolExecutor(max_workers=min(4, len(providers)))
        futures = {
            executor.submit(
                check_named_search_provider_connectivity,
                candidate,
                provider_id=provider,
                timeout_s=10.0,
                opener=opener,
            ): provider
            for provider in providers
        }
        done, pending = wait(futures, timeout=12.0)
        results: dict[str, JsonDict] = {}
        for future in done:
            provider = futures[future]
            try:
                results[provider] = dict(future.result())
            except Exception:  # noqa: BLE001 - Save returns a typed soft failure
                results[provider] = {
                    "provider": provider,
                    "ok": False,
                    "status": "network_error",
                    "probe_state": "network_unreachable",
                    "credential_state": "unknown",
                    "availability": "temporarily_unavailable",
                    "retry_after": time.time() + 60.0,
                    "reason_codes": ["provider_probe_failed"],
                    "error_message": "Provider probe failed.",
                }
        for future in pending:
            provider = futures[future]
            future.cancel()
            results[provider] = {
                "provider": provider,
                "ok": False,
                "status": "network_error",
                "probe_state": "network_unreachable",
                "credential_state": "unknown",
                "availability": "temporarily_unavailable",
                "retry_after": time.time() + 60.0,
                "reason_codes": ["provider_probe_batch_timeout"],
                "error_message": "Provider probe timed out.",
            }
        executor.shutdown(wait=False, cancel_futures=True)
        return results

    @staticmethod
    def _search_provider_conflict(expected: int, actual: int) -> BaselineError:
        return BaselineError(
            ErrorCode.SEARCH_PROVIDER_STATE_CONFLICT,
            "Search provider settings changed. Refresh and retry.",
            details={
                "operation_status": "conflict",
                "expected_revision": expected,
                "actual_revision": actual,
            },
        )

    # --- LLM registry: status / models / connection tests --------------------

    def llm_registry_status(self) -> JsonDict:
        """Return the secret-free LLM registry snapshot (profiles + role bindings)."""

        return self._registry.status()

    def upsert_llm_provider(self, payload: JsonDict) -> JsonDict:
        """Add or update an LLM provider profile; persist + rebuild; secret-free status.

        A blank/placeholder ``api_key`` preserves the existing stored key (so the UI
        never has to re-send secrets just to toggle ``enabled`` or edit models).
        """

        if not isinstance(payload, dict):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "provider payload must be an object")
        pid = str(payload.get("id") or "").strip()
        if not pid:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "provider id is required")
        previous_memory_ref = self._memory_provider_ref()
        existing = self._registry.get(pid)
        existing_signature = self._provider_probe_signature(existing)
        api_key = str(payload.get("api_key") or "")
        if not _configured_value(api_key) and existing is not None:
            api_key = existing.api_key
        models = payload.get("models")
        if isinstance(models, list):
            # Pass raw entries (str or rich dict) through; LLMProviderProfile coerces
            # each to a ModelProfile and drops empty ids.
            model_tuple = tuple(models)
        elif existing is not None:
            model_tuple = existing.models
        else:
            model_tuple = ()
        profile = LLMProviderProfile(
            id=pid,
            label=str(payload.get("label") if payload.get("label") is not None else (existing.label if existing else "")).strip(),
            type=str(payload.get("type") or (existing.type if existing else "openai")).strip() or "openai",
            base_url=str(payload.get("base_url") if payload.get("base_url") is not None else (existing.base_url if existing else "")).strip(),
            api_key=api_key,
            models=model_tuple,
            enabled=bool(payload.get("enabled", existing.enabled if existing else True)),
        )
        candidate = self._registry.upsert(profile)
        self._commit_registry(candidate)
        if self._memory_ref_targets_provider(pid, previous_memory_ref) and existing_signature != self._provider_probe_signature(profile):
            self._mark_memory_probe_stale("memory_provider_changed")
        self._wire_runtime()
        return self._registry.status()

    def remove_llm_provider(self, provider_id: str) -> JsonDict:
        """Remove a provider profile (scrubbing its role bindings); idempotent."""

        pid = str(provider_id or "").strip()
        if not pid:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "provider id is required")
        previous_memory_ref = self._memory_provider_ref()
        candidate = self._registry.remove(pid)
        self._commit_registry(candidate)
        if self._memory_ref_targets_provider(pid, previous_memory_ref):
            self._mark_memory_probe_stale("memory_provider_removed")
        self._wire_runtime()
        return self._registry.status()

    # --- LLM provider: partial update / key reveal / per-provider model CRUD --

    # Provider fields that a partial update (PATCH) may change. ``api_key`` is
    # handled separately so a blank value preserves the stored secret.
    _PROVIDER_PATCH_FIELDS = ("label", "type", "base_url", "enabled")

    def patch_llm_provider(self, provider_id: str, payload: JsonDict) -> JsonDict:
        """Partially update a saved provider (name/baseUrl/type/enabled/apiKey).

        Only the supplied fields change; a blank/placeholder ``api_key`` keeps the
        existing stored key. Models are never touched here (use the model CRUD).
        """

        if not isinstance(payload, dict):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "provider payload must be an object")
        pid = str(provider_id or "").strip()
        existing = self._registry.get(pid)
        if existing is None:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "unknown provider", details={"provider_id": pid})
        existing_signature = self._provider_probe_signature(existing)
        changes: JsonDict = {}
        if "label" in payload and payload["label"] is not None:
            changes["label"] = str(payload["label"]).strip()
        if "type" in payload and payload["type"] is not None:
            changes["type"] = str(payload["type"]).strip() or existing.type
        if "base_url" in payload and payload["base_url"] is not None:
            changes["base_url"] = str(payload["base_url"]).strip()
        if "enabled" in payload:
            changes["enabled"] = bool(payload["enabled"])
        if "api_key" in payload and _configured_value(str(payload.get("api_key") or "")):
            changes["api_key"] = str(payload["api_key"])
        profile = replace(existing, **changes)
        should_stale_memory_probe = (
            self._memory_ref_targets_provider(pid)
            and existing_signature != self._provider_probe_signature(profile)
        )
        candidate = self._registry.upsert(profile)
        self._commit_registry(candidate)
        if should_stale_memory_probe:
            self._mark_memory_probe_stale("memory_provider_changed")
        self._wire_runtime()
        return self._registry.status()

    def list_llm_provider_models(self, provider_id: str) -> JsonDict:
        """Return one provider's saved model profiles (secret-free)."""

        pid = str(provider_id or "").strip()
        profile = self._registry.get(pid)
        if profile is None:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "unknown provider", details={"provider_id": pid})
        return {"id": pid, "models": [m.status() for m in profile.models]}

    def create_llm_provider_models(self, provider_id: str, payload: JsonDict) -> JsonDict:
        """Add one or more models to a provider (batch); skips entries without an id.

        Accepts ``{"model": {...}}`` / ``{"id": "..."}`` for a single entry or
        ``{"models": [...]}`` for a batch. Existing model ids are replaced.
        """

        pid = str(provider_id or "").strip()
        if self._registry.get(pid) is None:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "unknown provider", details={"provider_id": pid})
        data = payload if isinstance(payload, dict) else {}
        raw_models = data.get("models")
        if isinstance(raw_models, list):
            entries = raw_models
        elif isinstance(data.get("model"), dict):
            entries = [data["model"]]
        else:
            entries = [data]
        added = 0
        should_stale_memory_probe = False
        candidate = self._registry
        for entry in entries:
            model = ModelProfile.from_config(entry)
            if not model.id:
                continue
            should_stale_memory_probe = should_stale_memory_probe or self._memory_ref_targets_model(pid, model.id)
            candidate = candidate.upsert_model(pid, model)
            added += 1
        if not added:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "no model with a valid id was provided")
        self._commit_registry(candidate)
        if should_stale_memory_probe:
            self._mark_memory_probe_stale("memory_model_changed")
        self._wire_runtime()
        return {"id": pid, "added": added, "models": [m.status() for m in self._registry.get(pid).models]}

    def update_llm_provider_model(self, provider_id: str, model_id: str, payload: JsonDict) -> JsonDict:
        """Partially update one model under a provider (cropped ``ModelProfile`` fields)."""

        pid = str(provider_id or "").strip()
        profile = self._registry.get(pid)
        if profile is None:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "unknown provider", details={"provider_id": pid})
        mid = str(model_id or "").strip()
        existing = profile.get_model(mid)
        if existing is None:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "unknown model", details={"model_id": mid})
        data = payload if isinstance(payload, dict) else {}
        # Merge onto the existing profile, keeping the model id stable.
        merged = {**existing.to_config(), **{k: v for k, v in data.items() if k != "id"}, "id": mid}
        should_stale_memory_probe = self._memory_ref_targets_model(pid, mid)
        candidate = self._registry.upsert_model(pid, ModelProfile.from_config(merged))
        self._commit_registry(candidate)
        if should_stale_memory_probe:
            self._mark_memory_probe_stale("memory_model_changed")
        self._wire_runtime()
        return {"id": pid, "models": [m.status() for m in self._registry.get(pid).models]}

    def delete_llm_provider_model(self, provider_id: str, model_id: str) -> JsonDict:
        """Remove one model from a provider (scrubbing any role refs); idempotent."""

        pid = str(provider_id or "").strip()
        if self._registry.get(pid) is None:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "unknown provider", details={"provider_id": pid})
        previous_memory_ref = self._memory_provider_ref()
        try:
            candidate = self._registry.remove_model(pid, model_id)
        except ValueError as error:
            raise BaselineError(ErrorCode.INVALID_REQUEST, str(error)) from error
        self._commit_registry(candidate)
        if self._memory_ref_targets_model(pid, model_id, previous_memory_ref):
            self._mark_memory_probe_stale("memory_model_removed")
        self._wire_runtime()
        return self._registry.status()

    def set_llm_roles(self, payload: JsonDict) -> JsonDict:
        """Set role bindings (default / embed / memory / council)."""

        if not isinstance(payload, dict):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "roles payload must be an object")
        council = payload.get("council")
        if council is not None and not isinstance(council, list):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "council must be a list of refs")
        previous_memory_ref = self._memory_provider_ref()
        try:
            candidate = self._registry.set_roles(
                default=payload.get("default"),
                embed=payload.get("embed"),
                memory=payload.get("memory"),
                council=council,
            )
        except ValueError as error:
            raise BaselineError(ErrorCode.INVALID_REQUEST, str(error)) from error
        self._commit_registry(candidate)
        self._mark_memory_probe_stale_if_ref_changed(previous_memory_ref, reason="memory_role_changed")
        self._wire_runtime()
        return self._registry.status()

    def _commit_registry(self, candidate: LLMRegistry) -> None:
        """Persist a candidate registry before exposing it to the running process."""

        if self._config_path is not None:
            save_llm_registry(self._config_path, candidate)
        self._registry = candidate

    def list_provider_models(self, payload: JsonDict, *, opener=None) -> JsonDict:
        """List available models for a candidate or saved LLM provider (secret-free).

        Accepts ``{type, base_url, api_key}`` for an ad-hoc candidate, or
        ``{provider_id}`` to reuse a saved profile's stored credentials (any
        explicitly supplied candidate field overrides the saved one).
        """

        ptype, base_url, api_key = self._resolve_llm_candidate(payload)
        if not base_url:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "base_url (or a configured provider_id) is required")
        result = list_models(type=ptype or "openai", base_url=base_url, api_key=api_key, opener=opener)
        return result.to_payload()

    def test_llm_provider(self, payload: JsonDict, *, opener=None) -> JsonDict:
        """Live-test an LLM provider or a single model (secret-free result).

        Two probe modes share this endpoint (SRB-aligned):

        - **Provider level (default)**: ``GET {base}/models`` reachability that
          does not depend on any configured model id. This is what the web
          Provider-row Test uses.
        - **Model level** (``{"probe": "chat", "model": "..."}``): a minimal
          chat-completions request to verify a specific model can infer. Used by
          the Models dialog row "Test chat" action.

        Ollama always uses its tags probe (it is not OpenAI-chat compatible).
        """

        ptype, base_url, api_key = self._resolve_llm_candidate(payload)
        data = payload if isinstance(payload, dict) else {}
        probe = str(data.get("probe") or "").strip().lower()
        explicit_model = str(data.get("model") or "").strip()

        if (ptype or "openai") == "ollama":
            # Ollama: tags probe doubles as both provider- and model-level test.
            result = list_models(type="ollama", base_url=base_url, api_key="", opener=opener)
            return {
                "ok": result.ok,
                "status": "reachable" if result.ok else result.status,
                "latency_ms": result.latency_ms,
                "model": explicit_model,
                "endpoint_host": result.endpoint_host,
                "error_message": result.error_message,
                "models_count": len(result.models) if result.ok else None,
            }

        if probe == "chat":
            # Model-level chat probe: requires an explicit model id.
            settings = ProviderSettings(llm_base_url=base_url, llm_api_key=api_key, llm_model=explicit_model)
            result = check_llm_provider_connectivity(settings, enabled=True, opener=opener)
            payload_out = result.to_payload()
            hint = _volcengine_base_hint(base_url)
            if hint and not result.ok:
                payload_out["error_message"] = _append_probe_hint(
                    str(payload_out.get("error_message") or ""), hint
                )
            return payload_out

        # Provider-level probe (default): GET models, no model dependency.
        return check_provider_models_reachable(
            type=ptype or "openai", base_url=base_url, api_key=api_key, opener=opener
        ).to_payload()

    def test_search_tools(self, payload: JsonDict, *, opener=None) -> JsonDict:
        """Live-test one search provider, or the legacy Brave+Firecrawl aggregate."""

        data = payload if isinstance(payload, dict) else {}
        brave = str(data.get("brave_api_key") or "").strip() or self._provider.brave_api_key
        firecrawl = str(data.get("firecrawl_api_key") or "").strip() or self._provider.firecrawl_api_key
        firecrawl_base = (
            str(data.get("firecrawl_base_url") or "").strip()
            or self._provider.firecrawl_base_url
            or "https://api.firecrawl.dev"
        )
        probe = ProviderSettings(
            brave_api_key=brave,
            firecrawl_api_key=firecrawl,
            firecrawl_base_url=firecrawl_base,
            exa_api_key=str(data.get("exa_api_key") or "").strip() or self._provider.exa_api_key,
            tavily_api_key=str(data.get("tavily_api_key") or "").strip() or self._provider.tavily_api_key,
        )
        selected = str(data.get("provider") or "").strip().lower()
        if selected:
            try:
                return check_named_search_provider_connectivity(
                    probe,
                    provider_id=selected,
                    opener=opener,
                )
            except ValueError as error:
                raise BaselineError(ErrorCode.INVALID_REQUEST, str(error)) from error
        return check_search_provider_connectivity(probe, enabled=True, opener=opener).to_payload()

    def _assert_session_accessible(self, session_id: str) -> None:
        """Raise SESSION_ARCHIVED when the session is archived (content locked)."""

        sid = str(session_id or "").strip()
        if not sid:
            return
        if self._store.is_session_archived(sid):
            raise BaselineError(
                ErrorCode.SESSION_ARCHIVED,
                "session is archived",
                details={"session_id": sid},
            )

    def _resolve_llm_candidate(self, payload: JsonDict) -> tuple[str, str, str]:
        """Return ``(type, base_url, api_key)`` from a candidate payload or saved profile."""

        data = payload if isinstance(payload, dict) else {}
        provider_id = str(data.get("provider_id") or "").strip()
        ptype = str(data.get("type") or "").strip()
        base_url = str(data.get("base_url") or "").strip()
        api_key = str(data.get("api_key") or "")
        if provider_id:
            profile = self._registry.get(provider_id)
            if profile is None:
                raise BaselineError(
                    ErrorCode.INVALID_REQUEST, "unknown provider", details={"provider_id": provider_id}
                )
            if _configured_value(api_key):
                # Ad-hoc credentials supplied: the caller fully controls the target.
                ptype = ptype or profile.type
                base_url = base_url or profile.base_url
            else:
                # Reuse the stored key ONLY against the saved profile's own host/type;
                # never send a saved BYOK key to a caller-overridden base_url.
                ptype = profile.type
                base_url = profile.base_url
                api_key = profile.api_key
        return ptype, base_url, api_key

    def index_url(self, payload: JsonDict) -> JsonDict:
        """Index a single URL into the local store (requires search adapter)."""

        url = str(payload.get("url") or "").strip()
        if not url:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "url is required")
        if not self._search:
            raise BaselineError(ErrorCode.PROVIDER_NOT_CONFIGURED, "search/fetch provider not configured")
        deletion_marker = self._store.get_document_deletion_marker(url)
        document = self._search.fetch(url)
        stats = self._indexer.index_document(
            document,
            revive_deleted_identity=True,
            expected_deletion_marker=deletion_marker,
        )
        if stats.get("skipped_deleted"):
            return stats
        document_id = str(stats["document_id"])
        try:
            self._store.add_library_provenance_for_current_document(
                artifact_type="source",
                artifact_id=document_id,
                document_id=document_id,
                expected_url=document.url,
                url=document.url,
                metadata={"source": "manual_index"},
            )
            self.capture_document_snapshot(document_id=document_id)
        except Exception:  # noqa: BLE001 - indexing succeeds even if snapshots fail
            pass
        return stats

    def list_documents(self) -> JsonDict:
        """List indexed library documents with chunk counts."""

        return {"documents": self._store.list_documents()}

    def rebuild_index(self, payload: JsonDict) -> JsonDict:
        """Synchronously rebuild chunks for already stored documents."""

        scope = str(payload.get("scope") or "all").strip().lower()
        if scope not in {"all", "documents"}:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "scope must be 'all' or 'documents'",
                details={"allowed": ["all", "documents"]},
            )
        if "url" in payload or "urls" in payload:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "index rebuild only accepts stored document ids")

        document_ids: list[str] | None = None
        if scope == "documents":
            raw_ids = payload.get("document_ids")
            if not isinstance(raw_ids, list) or not raw_ids:
                raise BaselineError(ErrorCode.INVALID_REQUEST, "document_ids is required for documents scope")
            document_ids = [str(item).strip() for item in raw_ids if str(item).strip()]
            if len(document_ids) != len(raw_ids):
                raise BaselineError(ErrorCode.INVALID_REQUEST, "document_ids must be non-empty strings")
            if len(document_ids) > 50:
                raise BaselineError(ErrorCode.INVALID_REQUEST, "document_ids may include at most 50 ids")

        documents = self._store.iter_documents_with_text(document_ids=document_ids)
        if document_ids is not None:
            found = {str(document.get("id") or "") for document in documents}
            missing = [document_id for document_id in document_ids if document_id not in found]
            if missing:
                raise BaselineError(
                    ErrorCode.INVALID_REQUEST,
                    "document not found",
                    details={"document_id": missing[0]},
                )

        rebuilt = 0
        chunks = 0
        for row in documents:
            metadata = row.get("metadata")
            if isinstance(metadata, str):
                try:
                    parsed = json.loads(metadata)
                except json.JSONDecodeError:
                    parsed = {}
                metadata = parsed if isinstance(parsed, dict) else {}
            elif not isinstance(metadata, dict):
                metadata = {}
            document = Document(
                url=str(row.get("url") or ""),
                title=str(row.get("title") or row.get("url") or ""),
                text=str(row.get("text") or ""),
                content_type=str(row.get("content_type") or "web"),
                metadata=metadata,
            )
            prepared = self._indexer.prepare_document(
                document,
                include_embeddings=str(metadata.get("source") or "") != "local_import",
            )
            stats = self._indexer.replace_prepared_document_if_present(
                prepared,
                expected_document_id=str(row.get("id") or ""),
            )
            if stats is None:
                continue
            rebuilt += 1
            chunks += int(stats.get("chunks") or 0)
        self._vector_backend.invalidate()
        return {"job_id": None, "status": "completed", "documents": rebuilt, "chunks": chunks, "errors": []}

    def capture_document_snapshot(self, *, document_id: str, run_id: str | None = None) -> JsonDict:
        """Capture a safe text snapshot from an already indexed document."""

        document = self._store.get_document(document_id)
        if not document:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "document not found")
        with self._snapshot_writer.operation_lock():
            written = self._snapshot_writer.write_text_snapshot(
                url=str(document.get("url") or ""),
                title=str(document.get("title") or ""),
                text=str(document.get("text") or ""),
            )
            try:
                snapshot_id = self._store.add_page_snapshot_with_provenance_if_current(
                    document_id=document_id,
                    expected_url=str(document.get("url") or ""),
                    url=written.url,
                    content_hash=written.content_hash,
                    storage_path=written.storage_path,
                    mime_type=written.mime_type,
                    size_bytes=written.size_bytes,
                    status="ok",
                    run_id=run_id,
                    metadata={"source": "library_snapshot"},
                )
            except Exception:
                self._snapshot_writer.delete_unreferenced(
                    [written.storage_path],
                    referenced_paths=set(self._store.list_snapshot_storage_paths()),
                )
                raise
        return {"snapshot_id": snapshot_id, "status": "ok"}

    # --- notes (user-authored, persisted as namespaced memories) --------------

    _NOTE_NAMESPACE = "note"
    _NOTE_SCOPE = "default"

    def add_note(self, payload: JsonDict) -> JsonDict:
        """Persist a free-text user note in the library."""

        content = str(payload.get("content") or "").strip()
        if not content:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "content is required")
        if len(content) > _MAX_TEXT_INPUT:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "content is too long")
        metadata = _safe_metadata(payload.get("metadata"))
        metadata.setdefault("source", "manual")
        note_id = self._store.add_memory(
            namespace=self._NOTE_NAMESPACE,
            scope_key=self._NOTE_SCOPE,
            content=content,
            metadata=metadata,
        )
        self._library_indexer.index_note(
            note_id=note_id,
            content=content,
            session_id=str(payload.get("session_id") or "") or None,
            run_id=str(payload.get("run_id") or "") or None,
            metadata=metadata,
        )
        return {"note_id": note_id}

    def list_notes(self) -> JsonDict:
        """List user notes newest-first."""

        notes = self._store.list_memories(
            namespace=self._NOTE_NAMESPACE, scope_key=self._NOTE_SCOPE, limit=200
        )
        notes = [self._with_note_provenance(note) for note in notes]
        return {"notes": notes}

    def _with_note_provenance(self, note: JsonDict) -> JsonDict:
        """Attach the latest run/session provenance to a note list row."""

        rows = self._store.list_library_provenance(
            artifact_type=self._NOTE_NAMESPACE,
            artifact_id=str(note.get("id") or ""),
        )
        if not rows:
            return note
        latest = rows[0]
        return {
            **note,
            "session_id": latest.get("session_id"),
            "run_id": latest.get("run_id"),
        }

    def list_library_documents(self) -> JsonDict:
        """List user-visible external and imported local Library sources."""

        return {"documents": self._store.list_external_documents()}

    def import_local_document(self, payload: JsonDict) -> JsonDict:
        """Parse and atomically index one local document without remote calls."""

        raw = payload.get("content")
        if not isinstance(raw, (bytes, bytearray)):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "file content is required")
        content = bytes(raw)
        if len(content) > MAX_UPLOAD_BYTES:
            raise BaselineError(
                ErrorCode.DOCUMENT_TOO_LARGE,
                "Document exceeds the 10 MiB import limit.",
                details={"max_bytes": MAX_UPLOAD_BYTES},
            )
        filename = sanitize_import_filename(str(payload.get("filename") or "document"))
        mime_type = str(payload.get("mime_type") or "")
        content_sha256 = hashlib.sha256(content).hexdigest()
        library_uri = f"library://local/{content_sha256}"
        deletion_marker = self._store.get_document_deletion_marker(library_uri)
        parsed = parse_local_document(
            content,
            filename=filename,
            mime_type=mime_type,
        )
        title = sanitize_import_title(
            str(payload.get("title") or ""),
            filename=filename,
        )
        prepared = self._indexer.prepare_document(
            Document(
                url=library_uri,
                title=title,
                text=parsed.text,
                content_type=parsed.content_type,
                metadata={
                    "source": "local_import",
                    "origin": "local",
                    "artifact_type": "library_document",
                    "filename": filename,
                    "content_sha256": content_sha256,
                },
            ),
            include_embeddings=False,
        )
        if not prepared.pieces:
            raise BaselineError(ErrorCode.DOCUMENT_NO_TEXT, "Document contains no extractable text.")
        try:
            stats = self._indexer.commit_local_import(
                prepared,
                content_sha256=content_sha256,
                filename=filename,
                expected_deletion_marker=deletion_marker,
            )
        except DocumentWriteRejected as error:
            raise BaselineError(
                ErrorCode.DOCUMENT_NOT_FOUND,
                "Document was deleted while import was in progress.",
            ) from error
        warnings = list(parsed.warnings)
        if prepared.sanitized.injections_removed:
            warnings.append("ingest_instructions_removed")
        if prepared.sanitized.markup_removed:
            warnings.append("ingest_markup_removed")
        if prepared.sanitized.truncated:
            warnings.append("ingest_text_truncated")
        if stats["status"] == "created":
            self._run_knowledge_graph_refresh_safe(workspace_id="default", run_id="")
        return {
            "status": stats["status"],
            "document": {
                "id": stats["document_id"],
                "title": stats["title"],
                "filename": stats["filename"],
                "content_type": stats["content_type"],
                "origin": "local",
                "artifact_type": "library_document",
                "content_sha256": stats["content_sha256"],
                "chunk_count": stats["chunks"],
            },
            "warnings": sorted(set(warnings)),
        }

    def delete_library_document(self, document_id: str) -> JsonDict:
        """Delete one authorized current Library Source and derived current state."""

        normalized_id = str(document_id or "").strip()
        if not normalized_id:
            raise BaselineError(ErrorCode.DOCUMENT_NOT_FOUND, "Document was not found.")
        deleted = self._store.delete_library_source_document(normalized_id)
        if deleted is None:
            raise BaselineError(ErrorCode.DOCUMENT_NOT_FOUND, "Document was not found.")
        if not deleted.get("deletable"):
            raise BaselineError(
                ErrorCode.DOCUMENT_NOT_DELETABLE,
                "This Library document cannot be deleted.",
            )

        warnings: list[str] = []
        try:
            with self._snapshot_writer.operation_lock():
                referenced_paths = set(self._store.list_snapshot_storage_paths())
                cleanup = self._snapshot_writer.delete_unreferenced(
                    deleted.get("snapshot_paths") or [],
                    referenced_paths=referenced_paths,
                )
                warnings.extend(str(item) for item in cleanup.get("warnings") or [])
                sweep = self._snapshot_writer.sweep_orphans(
                    referenced_paths=referenced_paths,
                    limit=4096,
                )
                warnings.extend(str(item) for item in sweep.get("warnings") or [])
        except Exception:  # noqa: BLE001 - database deletion remains authoritative
            warnings.append("snapshot_cleanup_failed")
        try:
            self._vector_backend.invalidate()
        except Exception:  # noqa: BLE001 - next query still falls back to lexical
            warnings.append("vector_invalidation_failed")
        self._run_knowledge_graph_refresh_safe(workspace_id="default", run_id="")
        return {
            "status": "deleted_with_cleanup_warning" if warnings else "deleted",
            "document_id": normalized_id,
            "origin": deleted.get("origin"),
            "artifact_type": deleted.get("artifact_type"),
            "chunks_deleted": int(deleted.get("chunks_deleted") or 0),
            "snapshots_deleted": int(deleted.get("snapshots_deleted") or 0),
            "provenance_deleted": int(deleted.get("provenance_deleted") or 0),
            "warnings": sorted(set(warnings)),
        }

    def list_library_notes(self) -> JsonDict:
        """List Library notes."""

        return self.list_notes()

    def add_library_note(self, payload: JsonDict) -> JsonDict:
        """Persist a Library note through the canonical Library route."""

        return self.add_note(payload)

    def get_note_provenance(self, note_id: str) -> JsonDict:
        """Return provenance rows for a Library note."""

        return {
            "provenance": self._store.list_library_provenance(
                artifact_type=self._NOTE_NAMESPACE,
                artifact_id=note_id,
            )
        }

    # --- reports and snapshots ------------------------------------------------

    def add_report(self, payload: JsonDict) -> JsonDict:
        """Persist and index a Library report."""

        title = str(payload.get("title") or "").strip()
        body_md = str(payload.get("body_md") or "").strip()
        mode = str(payload.get("mode") or "").strip()
        if not title:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "title is required")
        if not body_md:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "body_md is required")
        if not mode:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "mode is required")
        if len(body_md) > _MAX_REPORT_INPUT:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "body_md is too long")
        metadata = _safe_metadata(payload.get("metadata"))
        metadata.setdefault("source", "manual")
        parent_report_id = str(payload.get("parent_report_id") or "") or None
        session_id = str(payload.get("session_id") or "") or None
        run_id = str(payload.get("run_id") or "") or None
        report_id = self._store.add_report(
            title=title,
            body_md=body_md,
            mode=mode,
            parent_report_id=parent_report_id,
            session_id=session_id,
            run_id=run_id,
            metadata=metadata,
        )
        self._library_indexer.index_report(
            report_id=report_id,
            title=title,
            body_md=body_md,
            mode=mode,
            session_id=session_id,
            run_id=run_id,
            metadata=metadata,
        )
        return {"report_id": report_id}

    def upload_report(self, payload: JsonDict) -> JsonDict:
        """Persist a user-uploaded Markdown/text report."""

        filename = Path(str(payload.get("filename") or "report.md")).name
        suffix = Path(filename).suffix.lower()
        if suffix not in _UPLOAD_REPORT_EXTENSIONS:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "only .md and .txt reports can be uploaded")
        raw = payload.get("content")
        if not isinstance(raw, (bytes, bytearray)):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "file content is required")
        if len(raw) > _MAX_REPORT_INPUT:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "uploaded report is too large")
        try:
            decoded = bytes(raw).decode("utf-8")
        except UnicodeDecodeError as error:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "uploaded report must be UTF-8 text") from error
        sanitized = sanitize_ingest_text(decoded, max_chars=_MAX_REPORT_INPUT).text
        if not sanitized.strip():
            raise BaselineError(ErrorCode.INVALID_REQUEST, "uploaded report is empty")
        title = str(payload.get("title") or "").strip() or Path(filename).stem or "Uploaded report"
        return self.add_report(
            {
                "title": title,
                "body_md": sanitized,
                "mode": "upload",
                "metadata": {"source": "upload", "filename": filename},
            }
        )

    def list_library_reports(self) -> JsonDict:
        """List Library reports with child summaries attached to parents."""

        reports = self._store.list_reports(limit=200)
        children_by_parent: dict[str, list[JsonDict]] = {}
        roots: list[JsonDict] = []
        for report in reports:
            parent_id = report.get("parent_report_id")
            if parent_id:
                children_by_parent.setdefault(str(parent_id), []).append(report)
            else:
                roots.append(report)
        for report in roots:
            report["children"] = children_by_parent.get(str(report.get("id")), [])
        return {"reports": roots}

    def get_library_report(self, report_id: str) -> JsonDict:
        """Return one Library report with full Markdown and child summaries."""

        report = self._store.get_report(report_id)
        if not report:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "report not found")
        report["children"] = self._store.list_child_reports(report_id)
        return {"report": report}

    def backfill_library_index(self, payload: JsonDict | None = None) -> JsonDict:
        """Backfill missing derived Library index documents for stored artifacts."""

        payload = payload or {}
        requested = payload.get("artifact_types") or ["note", "report"]
        allowed = {str(item) for item in requested if str(item) in {"note", "report"}}
        if not allowed:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "artifact_types must include note or report")
        dry_run = bool(payload.get("dry_run", False))
        counts = {"note": 0, "report": 0, "memory": 0}

        if "note" in allowed:
            for note in self._store.list_memories(namespace=self._NOTE_NAMESPACE, scope_key=self._NOTE_SCOPE, limit=1000):
                if self._has_library_index("note", str(note.get("id") or "")):
                    continue
                counts["note"] += 1
                if not dry_run:
                    self._library_indexer.index_note(
                        note_id=str(note.get("id") or ""),
                        content=str(note.get("content") or ""),
                        metadata=note.get("metadata") if isinstance(note.get("metadata"), dict) else {},
                    )

        if "report" in allowed:
            for row in self._store.list_reports(limit=1000):
                report_id = str(row.get("id") or "")
                if self._has_library_index("report", report_id):
                    continue
                report = self._store.get_report(report_id) or {}
                counts["report"] += 1
                if not dry_run:
                    self._library_indexer.index_report(
                        report_id=report_id,
                        title=str(report.get("title") or ""),
                        body_md=str(report.get("body_md") or ""),
                        mode=str(report.get("mode") or ""),
                        session_id=str(report.get("session_id") or "") or None,
                        run_id=str(report.get("run_id") or "") or None,
                        metadata=report.get("metadata") if isinstance(report.get("metadata"), dict) else {},
                    )

        return {"dry_run": dry_run, "indexed": counts}

    def _has_library_index(self, artifact_type: str, artifact_id: str) -> bool:
        """Return whether an artifact already has derived Library provenance."""

        return bool(self._store.list_library_provenance(artifact_type=artifact_type, artifact_id=artifact_id))

    def _index_memory_for_library(
        self,
        *,
        memory_id: str,
        namespace: str,
        scope_key: str,
        content: str,
        metadata: JsonDict | None = None,
    ) -> JsonDict | None:
        """Compatibility no-op: local memories no longer enter Library."""

        return None

    def list_source_sessions(self, document_id: str) -> JsonDict:
        """Return session/run provenance for a source document."""

        rows = [
            *self._store.list_library_provenance(
                artifact_type="source",
                artifact_id=document_id,
            ),
            *self._store.list_library_provenance(
                artifact_type="library_document",
                artifact_id=document_id,
            ),
        ]
        rows.sort(key=lambda item: float(item.get("created_at") or 0), reverse=True)
        return {"artifacts": rows}

    def list_source_snapshots(self, document_id: str) -> JsonDict:
        """Return snapshot metadata for a source document without filesystem paths."""

        snapshots = [self._public_snapshot(row) for row in self._store.list_page_snapshots_for_document(document_id)]
        return {"snapshots": snapshots}

    def get_library_snapshot(self, snapshot_id: str) -> JsonDict:
        """Return a bounded text preview for one snapshot."""

        snapshot = self._store.get_page_snapshot(snapshot_id)
        if not snapshot:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "snapshot not found")
        public = self._public_snapshot(snapshot)
        if snapshot.get("status") == "ok":
            public["text"] = self._snapshot_writer.read_text(str(snapshot.get("storage_path") or ""))[
                :_SNAPSHOT_PREVIEW_CHARS
            ]
        return {"snapshot": public}

    def _public_snapshot(self, snapshot: JsonDict) -> JsonDict:
        """Strip local filesystem internals from snapshot metadata."""

        return {
            key: value
            for key, value in snapshot.items()
            if key not in {"storage_path", "content_hash", "metadata"}
        }

    def _persist_library_artifacts(self, request: RunRequest, outcome) -> None:  # noqa: ANN001
        """Best-effort post-run Library artifacts."""

        session_id = request.session_id
        run_id = outcome.run_id
        mode = request.mode.value
        metadata = {"source": "post_run"}
        title = _report_title(request.query, mode)

        parent_report_id: str | None = None
        if request.mode in {Mode.DEEP, Mode.COUNCIL}:
            try:
                parent_report_id = self.add_report(
                    {
                        "title": title,
                        "body_md": outcome.answer.text,
                        "mode": mode,
                        "session_id": session_id,
                        "run_id": run_id,
                        "metadata": metadata,
                    }
                )["report_id"]
            except Exception as error:  # noqa: BLE001 - post-run artifacts are best-effort
                self._emit_library_artifact_warning(run_id, stage="parent_report", error=error)
                parent_report_id = None

            if request.mode is Mode.COUNCIL and parent_report_id:
                for member in outcome.extra.get("members", []) or []:
                    try:
                        answer = member.get("answer", {}) if isinstance(member, dict) else {}
                        body_md = str(answer.get("text") or "").strip()
                        if not body_md:
                            continue
                        member_name = str(member.get("model") or "Council member")
                        self.add_report(
                            {
                                "title": member_name,
                                "body_md": body_md,
                                "mode": "council_member",
                                "parent_report_id": parent_report_id,
                                "session_id": session_id,
                                "run_id": run_id,
                                "metadata": {
                                    "source": "post_run",
                                    "member": member_name,
                                    "perspective": str(member.get("perspective") or ""),
                                },
                            }
                        )
                    except Exception as error:  # noqa: BLE001
                        self._emit_library_artifact_warning(run_id, stage="member_report", error=error)
                        continue

        note_content = _auto_note_content(outcome.answer)
        if note_content and mode in set(self._post_run_notes_config.enabled_modes):
            try:
                self.add_note(
                    {
                        "content": note_content,
                        "session_id": session_id,
                        "run_id": run_id,
                        "metadata": metadata,
                    }
                )
            except Exception as error:  # noqa: BLE001
                self._emit_library_artifact_warning(run_id, stage="post_run_note", error=error)
                pass

        if request.mode in {Mode.FAST, Mode.MULTIPLE, Mode.DEEP, Mode.COUNCIL}:
            self._capture_citation_snapshots(outcome, run_id=run_id)
        self._enqueue_memory_refresh_after_run(request, outcome)
        self._enqueue_knowledge_graph_refresh_after_run(request, outcome)

    def _enqueue_memory_refresh_after_run(self, request: RunRequest, outcome) -> None:  # noqa: ANN001
        """Schedule best-effort memory continuity refresh for usable terminal output."""

        if not self._should_refresh_memory_after_run(request, outcome):
            return
        payload = {
            "session_id": str(request.session_id or ""),
            "run_id": str(outcome.run_id or ""),
            "workspace_key": str(request.workspace_key or "default"),
            "user_key": str(request.user_key or "default"),
        }
        try:
            if self._job_backend.name == "rq":
                self._store.upsert_post_turn_memory_run(
                    **payload,
                    status="queued",
                    provider_ref=self._memory_provider_ref(),
                )
                self._job_backend.enqueue_post_turn_memory(payload)
                return
            self._memory_refresh_queue.enqueue(
                session_id=str(request.session_id or ""),
                run_id=str(outcome.run_id or ""),
                workspace_key=str(request.workspace_key or "default"),
                user_key=str(request.user_key or "default"),
            )
        except Exception as error:  # noqa: BLE001 - post-run refresh must never fail the run
            if self._job_backend.name == "rq":
                self.fail_post_turn_memory_job(payload, error_code=error.__class__.__name__)
            self._emit_library_artifact_warning(str(outcome.run_id or ""), stage="memory_refresh", error=error)

    def _should_refresh_memory_after_run(self, request: RunRequest, outcome) -> bool:  # noqa: ANN001
        if not self._memory_settings.post_turn_memory_refresh_enabled:
            return False
        probe = self._memory_settings.format_probe if isinstance(self._memory_settings.format_probe, dict) else {}
        if probe.get("stale") is True or str(probe.get("status") or "") == "failed":
            return False
        return self._has_usable_terminal_output(request, outcome)

    def _has_usable_terminal_output(self, request: RunRequest, outcome) -> bool:  # noqa: ANN001
        if not str(request.session_id or "").strip():
            return False
        status = str(getattr(outcome, "status", "") or "completed")
        text = str(getattr(getattr(outcome, "answer", None), "text", "") or "").strip()
        if not text:
            return False
        if status in {"completed", "completed_degraded"}:
            return True
        if status == "partial_timeout":
            return True
        return False

    def _enqueue_knowledge_graph_refresh_after_run(self, request: RunRequest, outcome) -> None:  # noqa: ANN001
        """Schedule best-effort Knowledge Graph rebuild for usable terminal output."""

        if not self._should_refresh_knowledge_graph_after_run(request, outcome):
            return
        try:
            self._knowledge_graph_refresh_queue.enqueue(
                workspace_id=str(request.workspace_key or "default"),
                run_id=str(outcome.run_id or ""),
            )
        except Exception as error:  # noqa: BLE001 - post-run refresh must never fail the run
            self._emit_library_artifact_warning(
                str(outcome.run_id or ""),
                stage="knowledge_graph_refresh",
                error=error,
            )

    def _should_refresh_knowledge_graph_after_run(self, request: RunRequest, outcome) -> bool:  # noqa: ANN001
        if not self._knowledge_graph_config.post_run_rebuild_enabled:
            return False
        return self._has_usable_terminal_output(request, outcome)

    def _run_memory_refresh_safe(
        self,
        *,
        session_id: str,
        run_id: str,
        workspace_key: str,
        user_key: str,
    ) -> None:
        """Refresh memories after a run without affecting terminal run state."""

        try:
            self.refresh_post_turn_memory(
                session_id=session_id,
                run_id=run_id,
                workspace_key=workspace_key,
                user_key=user_key,
            )
        except Exception as error:  # noqa: BLE001 - refresh is observability-only on failure
            self._emit_library_artifact_warning(run_id, stage="memory_refresh", error=error)

    def execute_post_turn_memory_job(self, payload: JsonDict) -> JsonDict:
        """Execute one already-authorized post-turn refresh in maintenance RQ."""

        session_id = str(payload.get("session_id") or "")
        run_id = str(payload.get("run_id") or "")
        workspace_key = str(payload.get("workspace_key") or "default")
        user_key = str(payload.get("user_key") or "default")
        run = self._store.get_run(run_id)
        if (
            not run
            or str(run.get("session_id") or "") != session_id
            or str(run.get("status") or "")
            not in {"completed", "completed_degraded", "partial_timeout"}
        ):
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "post-turn Memory job does not match a delivered run",
            )
        provider_ref = self._memory_provider_ref()
        self._store.upsert_post_turn_memory_run(
            run_id=run_id,
            session_id=session_id,
            workspace_key=workspace_key,
            user_key=user_key,
            status="running",
            provider_ref=provider_ref,
        )
        try:
            result = self.refresh_post_turn_memory(
                session_id=session_id,
                run_id=run_id,
                workspace_key=workspace_key,
                user_key=user_key,
            )
            embedding_status = "not_configured"
            memory_id = str(result.get("memory_id") or "")
            memory = self._store.get_memory(memory_id) if memory_id else None
            embedder = self._resolve_semantic_embedder()
            if embedder is not None and memory is not None:
                content = str(memory.get("content") or "")
                if self._store.workspace_memory_has_embedding(memory_id, content):
                    embedding_status = "completed"
                else:
                    deadline = time.monotonic() + 15.0
                    embed_budget = ProviderCallBudget(
                        request_deadline=deadline,
                        phase_deadline=deadline,
                    )
                    with use_provider_call_budget(embed_budget):
                        vector = MemoryEngine(self._store, embedder=embedder)._embed_one(
                            content,
                            timeout_cap_s=15.0,
                        )
                    applied = bool(
                        vector
                        and self._store.compare_and_set_workspace_memory_embedding(
                            memory_id,
                            content,
                            vector,
                        )
                    )
                    embedding_status = (
                        "completed"
                        if applied
                        or self._store.workspace_memory_has_embedding(memory_id, content)
                        else "failed"
                    )
            self._store.upsert_post_turn_memory_run(
                run_id=run_id,
                session_id=session_id,
                workspace_key=workspace_key,
                user_key=user_key,
                status="completed",
                provider_ref=provider_ref,
                embedding_status=embedding_status,
            )
            return {**result, "embedding_status": embedding_status}
        except Exception as error:
            self.fail_post_turn_memory_job(
                payload,
                error_code=str(getattr(error, "error_code", error.__class__.__name__)),
            )
            raise

    def fail_post_turn_memory_job(self, payload: JsonDict, *, error_code: str) -> bool:
        """Persist a failed maintenance audit without changing report state/events."""

        run_id = str(payload.get("run_id") or "")
        session_id = str(payload.get("session_id") or "")
        run = self._store.get_run(run_id)
        if not run or str(run.get("session_id") or "") != session_id:
            return False
        return self._store.fail_post_turn_memory_run(
            run_id=run_id,
            session_id=session_id,
            error_code=str(error_code or "failed"),
        )

    def _run_knowledge_graph_refresh_safe(self, *, workspace_id: str, run_id: str) -> None:
        """Trigger a KG rebuild without affecting terminal run state."""

        try:
            self._knowledge_graph.start_rebuild({"workspace_id": workspace_id or "default"})
        except BaselineError as error:
            if error.code == ErrorCode.KNOWLEDGE_GRAPH_BUILD_RUNNING:
                return
            self._emit_library_artifact_warning(run_id, stage="knowledge_graph_refresh", error=error)
        except Exception as error:  # noqa: BLE001 - refresh is observability-only on failure
            self._emit_library_artifact_warning(run_id, stage="knowledge_graph_refresh", error=error)

    def _run_wiki_refresh_job_safe(self, payload: JsonDict, job_id: str) -> None:
        """Refresh Wiki Map in a thread backend without leaking failures to callers."""

        try:
            self.refresh_knowledge_graph_wiki_now(payload or {}, job_id=job_id)
        except BaselineError as error:
            self._wiki_store.mark_refresh_job_failed(
                job_id,
                error_code=error.code.value,
                error_message=error.message,
            )
        except Exception as error:  # noqa: BLE001
            self._wiki_store.mark_refresh_job_failed(
                job_id,
                error_code="INTERNAL_ERROR",
                error_message=_safe_status_error(str(error)),
            )

    def _on_knowledge_graph_rebuild_succeeded(self, result: JsonDict) -> None:
        """Mark Wiki stale after a new KG version and optionally enqueue refresh."""

        workspace_id = str(result.get("workspace_id") or "default").strip() or "default"
        self._wiki_store.mark_active_stale(workspace_id, reason="graph_version_changed")
        if not self._knowledge_graph_config.wiki_auto_refresh_after_kg_rebuild:
            return
        try:
            self.start_knowledge_graph_wiki_refresh({"workspace_id": workspace_id})
        except BaselineError as error:
            if error.code == ErrorCode.KNOWLEDGE_GRAPH_WIKI_REFRESH_RUNNING:
                return
            return
        except Exception:
            return

    def _capture_citation_snapshots(self, outcome, *, run_id: str) -> None:  # noqa: ANN001
        """Capture the exact adopted citation documents without another fetch.

        T49 materializes the final adopted evidence before the terminal
        transition.  Re-fetching after that transition would escape the absolute
        run deadline and could make the Library snapshot differ from the evidence
        that grounded the answer.
        """

        for citation in getattr(outcome.answer, "citations", []) or []:
            document_id = getattr(citation, "document_id", "")
            if not document_id:
                continue
            url = str(getattr(citation, "url", "") or "")
            artifact_type = str(getattr(citation, "artifact_type", "") or "")
            artifact_id = str(getattr(citation, "artifact_id", "") or document_id)
            snapshot_document_id = document_id
            try:
                current_document = self._store.get_document(snapshot_document_id)
                if not current_document:
                    raise DocumentWriteRejected("cited document no longer exists")
                expected_url = str(current_document.get("url") or "")
                if artifact_type == "library_document":
                    self._store.add_library_provenance_for_current_document(
                        artifact_type="library_document",
                        artifact_id=artifact_id,
                        document_id=snapshot_document_id,
                        expected_url=expected_url,
                        run_id=run_id,
                        metadata={"source": "post_run_citation", "origin": "local"},
                    )
                    continue
                self._store.add_library_provenance_for_current_document(
                    artifact_type="source",
                    artifact_id=snapshot_document_id,
                    document_id=snapshot_document_id,
                    expected_url=expected_url,
                    run_id=run_id,
                    url=url or None,
                    metadata={"source": "post_run_citation"},
                )
                self.capture_document_snapshot(document_id=snapshot_document_id, run_id=run_id)
            except Exception as error:  # noqa: BLE001
                self._emit_library_artifact_warning(run_id, stage="citation_snapshot", error=error)
                continue

    def _emit_library_artifact_warning(self, run_id: str, *, stage: str, error: Exception) -> None:
        """Emit a non-fatal post-run artifact warning."""

        try:
            self._store.append_event(
                run_id,
                "library_artifact_warning",
                {"stage": stage, "message": _safe_status_error(str(error) or error.__class__.__name__)},
            )
        except Exception:  # noqa: BLE001 - warning observability must never become fatal
            return

    # --- tools ----------------------------------------------------------------

    def enhance_prompt(self, payload: JsonDict) -> JsonDict:
        """Rewrite a raw query into a clearer, well-scoped research prompt via the LLM."""

        query = str(payload.get("query") or "").strip()
        if not query:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "query is required")
        if len(query) > _MAX_TEXT_INPUT:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "query is too long")
        if not self._llm_ready:
            raise BaselineError(
                ErrorCode.PROVIDER_NOT_CONFIGURED,
                "No LLM backend configured: set BYOK provider keys or start a local Ollama server.",
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "You rewrite a user's research request into a single, clear, well-scoped "
                    "research question. Keep the user's intent and language. Return only the "
                    "rewritten question with no preamble, quotes, or commentary."
                ),
            },
            {"role": "user", "content": query},
        ]
        result = self._llm.complete(messages, temperature=0.3, max_tokens=200)
        enhanced = (result.text or "").strip() or query
        return {"enhanced": enhanced}

    def _provider_status(self) -> JsonDict:
        status = self._provider.safe_status()
        states = self._search_provider_states()
        capabilities = provider_capabilities(states)
        return {
            "llm_base_url": self._provider.llm_base_url if status["llm_base_url"] else "",
            "llm_model": self._provider.llm_model if status["llm_model"] else "",
            "has_llm_key": status["llm_api_key"],
            "brave_ready": status["brave_api_key"],
            "firecrawl_ready": status["firecrawl_api_key"],
            "exa_ready": status["exa_api_key"],
            "tavily_ready": status["tavily_api_key"],
            "search_providers": capabilities,
            "provider_config_revision": int(
                self._search_provider_state.get("revision") or 0
            ),
            "external_search_service": external_search_service_status(states),
            "embed_model": "",
            "ollama_base_url": "",
        }


def _dream_status_payload(row: JsonDict) -> JsonDict:
    """Return a secret-free Dreaming status payload."""

    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    payload: JsonDict = {
        "status": str(row.get("status") or "idle"),
        "dream_run_id": row.get("id"),
        "workspace_key": row.get("workspace_key") or "default",
        "user_key": row.get("user_key") or "default",
        "started_at": row.get("started_at"),
        "finished_at": row.get("finished_at"),
        "updated_at": row.get("updated_at"),
        "cancel_requested_at": row.get("cancel_requested_at"),
        "backend": row.get("backend") or "thread",
        "backend_job_id": row.get("backend_job_id"),
        "memories_created": int(row.get("memories_created") or 0),
        "error": row.get("error"),
        "metadata": metadata,
    }
    for key in (
        "error_code",
        "failed_stage",
        "failed_session_id",
        "attempted_sessions",
        "successful_sessions",
        "failed_sessions",
        "workspace_skipped",
        "user_profile_attempted",
        "user_profile_updated",
        "user_profile_failed",
        "failure_count",
        "provider_ref",
        "structured_strategy",
        "response_shape",
        "finish_reason",
        "validation_error",
        "incomplete",
        "incomplete_reasons",
        "workspace_memory_refresh_strategy",
        "include_facts",
        "accepted",
        "skipped",
        "failed",
        "fact_attempted",
        "fact_source",
    ):
        if key in metadata:
            payload[key] = metadata.get(key)
    return payload


def _public_session(row: JsonDict) -> JsonDict:
    """Normalize session preference fields for API clients."""

    public = {key: value for key, value in row.items() if key != "metadata"}
    policy = _session_search_usage_policy(row)
    return {
        **public,
        "pinned": bool(row.get("pinned")),
        "mode_default": row.get("mode_default") or None,
        "search_usage_policy": policy.to_dict() if policy is not None else None,
    }


def _session_search_usage_policy(row: JsonDict | None) -> SearchUsagePolicy | None:
    """Read a session override from private metadata without exposing siblings."""

    if not row:
        return None
    metadata = row.get("metadata")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise BaselineError(
                ErrorCode.INTERNAL_ERROR,
                "stored session metadata is invalid",
            ) from error
    if not isinstance(metadata, dict):
        raise BaselineError(ErrorCode.INTERNAL_ERROR, "stored session metadata is invalid")
    if "search_usage_policy" not in metadata:
        return None
    if not isinstance(metadata.get("search_usage_policy"), dict):
        raise BaselineError(ErrorCode.INTERNAL_ERROR, "stored session search policy is invalid")
    try:
        return normalize_search_usage_policy(metadata["search_usage_policy"], require_complete=True)
    except ValueError as error:
        raise BaselineError(
            ErrorCode.INTERNAL_ERROR,
            "stored session search policy is invalid",
        ) from error


def _validate_session_patch_payload(payload: JsonDict | None) -> JsonDict:
    if not isinstance(payload, dict) or not payload:
        raise BaselineError(ErrorCode.INVALID_REQUEST, "session patch payload is required")
    allowed = {"pinned", "mode_default", "search_usage_policy"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise BaselineError(ErrorCode.INVALID_REQUEST, "unsupported session patch field", details={"fields": unknown})
    updates: JsonDict = {}
    if "pinned" in payload:
        pinned = payload.get("pinned")
        if not isinstance(pinned, bool):
            raise BaselineError(ErrorCode.INVALID_REQUEST, "pinned must be a boolean")
        updates["pinned"] = pinned
    if "mode_default" in payload:
        mode_default = payload.get("mode_default")
        if mode_default is None:
            updates["mode_default"] = None
        else:
            mode = str(mode_default).lower()
            if mode not in {item.value for item in Mode}:
                raise BaselineError(ErrorCode.INVALID_REQUEST, "mode_default must be a valid mode or null")
            updates["mode_default"] = mode
    if "search_usage_policy" in payload:
        search_usage = payload.get("search_usage_policy")
        if search_usage is None:
            updates["search_usage_policy"] = None
        else:
            try:
                updates["search_usage_policy"] = normalize_search_usage_policy(
                    search_usage,
                    require_complete=True,
                )
            except ValueError as error:
                raise BaselineError(ErrorCode.INVALID_REQUEST, str(error)) from error
    if not updates:
        raise BaselineError(ErrorCode.INVALID_REQUEST, "session patch payload is required")
    return updates


def _audit_from_dream_status(status: JsonDict) -> JsonDict:
    """Normalize a Dream status payload into the Wave 1 write-audit DTO."""

    failed_sessions = int(status.get("failed_sessions") or 0)
    user_failed = 1 if bool(status.get("user_profile_failed")) else 0
    failed = int(status.get("failed") if status.get("failed") is not None else status.get("failure_count") or failed_sessions + user_failed)
    return {
        "status": str(status.get("status") or "idle"),
        "source": str(status.get("fact_source") or (status.get("metadata") or {}).get("source") or "memory_continuity_v1")
        if isinstance(status.get("metadata"), dict)
        else "memory_continuity_v1",
        "attempted": int(status.get("fact_attempted") if status.get("fact_attempted") is not None else status.get("attempted_sessions") or 0),
        "succeeded": int(status.get("accepted") if status.get("accepted") is not None else status.get("successful_sessions") or 0),
        "skipped": int(status.get("skipped") if status.get("skipped") is not None else status.get("workspace_skipped") or 0),
        "failed": failed,
        "error_code": status.get("error_code"),
        "checked_at": status.get("finished_at") or status.get("started_at"),
        "provider_ref": status.get("provider_ref") or "",
        "stale": False,
        "incomplete_reasons": status.get("incomplete_reasons") if isinstance(status.get("incomplete_reasons"), list) else [],
    }


def _dream_next_scheduled_at(row: JsonDict | None, schedule: str) -> float | None:
    """Return the next Dreaming timestamp for a schedule, or ``None`` when off/running."""

    if schedule == "off":
        return None
    if row and row.get("status") in _DREAM_ACTIVE_STATUSES:
        return None
    interval = _DREAM_SCHEDULE_INTERVALS.get(schedule)
    if interval is None:
        return None
    if not row:
        return 0.0
    base = row.get("finished_at") or row.get("started_at")
    return float(base or 0.0) + interval


def _normalize_wiki_backend_job_id(job_id: str) -> str:
    clean_job_id = str(job_id or "")
    for prefix in ("kg-wiki-refresh:", "kg-wiki-refresh-"):
        if clean_job_id.startswith(prefix):
            return clean_job_id[len(prefix) :]
    return clean_job_id


def _safe_status_error(message: str) -> str:
    """Sanitize a short status error for UI/API use."""

    text = re.sub(r"[A-Za-z]:\\[^\s]+", "[path]", message)
    text = re.sub(r"/[^\s]+", "[path]", text)
    text = _SECRET_VALUE_RE.sub("[redacted]", text)
    return text[:240]


def _sse(event_type: str, payload: JsonDict, *, event_id: int | None = None) -> bytes:
    prefix = f"id: {event_id}\n" if event_id is not None else ""
    return f"{prefix}event: {event_type}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


class _UnconfiguredLLM:
    """Placeholder LLM that raises until a provider is configured (satisfies LLMAdapter)."""

    name = "unconfigured"

    def complete(  # noqa: ANN001, ARG002
        self,
        messages,
        *,
        model=None,
        temperature=0.2,
        max_tokens=1200,
        thinking_budget=None,
        response_format=None,
    ):
        """Raise PROVIDER_NOT_CONFIGURED on any completion attempt."""

        raise BaselineError(ErrorCode.PROVIDER_NOT_CONFIGURED, "LLM provider is not configured")

    def embed(self, texts, *, model=None):  # noqa: ANN001, ARG002
        """Raise PROVIDER_NOT_CONFIGURED on any embedding attempt."""

        raise BaselineError(ErrorCode.PROVIDER_NOT_CONFIGURED, "LLM provider is not configured")
