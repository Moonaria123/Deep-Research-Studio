"""FastAPI gateway for the baseline subsystem (/api/v1).

FastAPI/Starlette are imported lazily inside :func:`create_app` so importing this module
never requires the web stack. Errors are normalized to the baseline error envelope:
``{"error": {"code": ..., "message": ..., "details": {...}}}``.
"""

from __future__ import annotations

import asyncio
import os
import secrets
from collections.abc import Collection
from typing import Any
from urllib.parse import urlsplit

from .errors import BaselineError, ErrorCode, error_payload
from .local_document_import import MAX_UPLOAD_BYTES
from .service import BaselineService

JsonDict = dict[str, Any]
API_PREFIX = "/api/v1"

# Endpoints reachable without a bearer token even in ``token`` auth mode. ``/health`` must
# stay open so Docker healthchecks and sidecar readiness probes work without credentials.
_PUBLIC_PATHS = frozenset({f"{API_PREFIX}/health"})


async def _read_upload_bounded(upload: Any, *, max_bytes: int) -> bytes:
    """Read one upload incrementally and fail as soon as the byte ceiling is crossed."""

    body = bytearray()
    while True:
        chunk = await upload.read(min(64 * 1024, max_bytes + 1 - len(body)))
        if not chunk:
            break
        body.extend(chunk)
        if len(body) > max_bytes:
            raise BaselineError(
                ErrorCode.DOCUMENT_TOO_LARGE,
                "Document exceeds the 10 MiB import limit.",
                details={"max_bytes": max_bytes},
            )
    return bytes(body)


def _parse_job_event_cursor(
    after_seq: str | None,
    last_event_id: str | None,
) -> int:
    """Parse the explicit query cursor before the SSE header cursor."""

    raw: object = after_seq if after_seq is not None else (last_event_id or "0")
    try:
        cursor = int(raw)
    except (TypeError, ValueError, OverflowError) as error:
        raise BaselineError(
            ErrorCode.INVALID_REQUEST,
            "after_seq/Last-Event-ID must be a non-negative integer",
        ) from error
    if cursor < 0 or cursor > (2**63 - 1):
        raise BaselineError(
            ErrorCode.INVALID_REQUEST,
            "after_seq/Last-Event-ID must be a non-negative integer",
        )
    return cursor


def _resolve_auth(auth_mode: str | None, auth_token: str | None) -> tuple[str, str]:
    """Resolve the effective auth mode + token, falling back to env, and fail fast on misconfig.

    ``DEEPRESEARCH_AUTH_MODE`` defaults to ``off`` so localhost/dev keeps zero-friction access
    (DR-ITER-027). In ``token`` mode a non-empty ``DEEPRESEARCH_AUTH_TOKEN`` is mandatory: we
    refuse to start an "authenticated" server with no token rather than silently accepting all
    requests.
    """

    mode = (auth_mode if auth_mode is not None else os.environ.get("DEEPRESEARCH_AUTH_MODE", "off"))
    mode = (mode or "off").strip().lower()
    token = (auth_token if auth_token is not None else os.environ.get("DEEPRESEARCH_AUTH_TOKEN", ""))
    token = (token or "").strip()

    if mode not in {"off", "token"}:
        raise BaselineError(
            ErrorCode.INVALID_REQUEST,
            f"DEEPRESEARCH_AUTH_MODE must be 'off' or 'token', got {mode!r}",
        )
    if mode == "token" and not token:
        raise BaselineError(
            ErrorCode.INVALID_REQUEST,
            "DEEPRESEARCH_AUTH_MODE=token requires a non-empty DEEPRESEARCH_AUTH_TOKEN.",
        )
    return mode, token


def _validate_cors_origins(cors_origins: Collection[str] | None) -> tuple[str, ...]:
    """Return safe, exact CORS origins or fail fast on an unsafe value."""

    validated: list[str] = []
    for raw_origin in cors_origins or ():
        origin = str(raw_origin or "").strip()
        invalid = BaselineError(
            ErrorCode.INVALID_REQUEST,
            "cors_origins must contain exact http(s) origins without paths or credentials.",
        )
        try:
            parsed = urlsplit(origin)
            # Accessing ``port`` performs urllib's numeric/range validation.
            parsed.port
        except ValueError:
            raise invalid from None
        if (
            origin == "*"
            or parsed.scheme not in {"http", "https"}
            or not parsed.netloc
            or not parsed.hostname
            or parsed.path
            or parsed.query
            or parsed.fragment
            or parsed.username is not None
            or parsed.password is not None
        ):
            raise invalid
        if origin not in validated:
            validated.append(origin)
    return tuple(validated)


def create_app(
    service: BaselineService,
    *,
    title: str = "DeepResearch API",
    auth_mode: str | None = None,
    auth_token: str | None = None,
    cors_origins: Collection[str] | None = None,
):
    """Build a FastAPI app exposing the baseline endpoints over ``service``.

    ``auth_mode``/``auth_token`` default to the ``DEEPRESEARCH_AUTH_MODE`` /
    ``DEEPRESEARCH_AUTH_TOKEN`` environment variables. When mode is ``token`` every
    ``/api/v1/*`` route except ``/health`` requires an ``Authorization: Bearer <token>`` header.
    CORS is disabled unless ``cors_origins`` supplies exact HTTP(S) origins; credentialed
    cross-origin requests are never enabled.
    """

    mode, token = _resolve_auth(auth_mode, auth_token)
    allowed_cors_origins = _validate_cors_origins(cors_origins)

    try:
        from fastapi import Body, FastAPI, File, Form, Request, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse, StreamingResponse
        globals()["UploadFile"] = UploadFile
        globals()["Request"] = Request
    except Exception as error:  # noqa: BLE001
        raise BaselineError(
            ErrorCode.PROVIDER_NOT_CONFIGURED,
            "fastapi is required to create the API app (pip install fastapi uvicorn)",
        ) from error

    app = FastAPI(title=title, version="1.0.0")
    app.router.add_event_handler("shutdown", service.shutdown)

    # Loopback publishing alone does not stop DNS rebinding or cross-site writes
    # from a browser on this machine. The local web distribution opts into both
    # Host validation and exact browser-origin checks, including read endpoints.
    if os.environ.get("DEEPRESEARCH_LOCAL_WEB_GUARD") == "1":
        web_origins = _validate_cors_origins(
            tuple(filter(None, os.environ.get("DEEPRESEARCH_WEB_ORIGINS", "").split(",")))
        )
        if not web_origins:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "Local web guard requires exact web origins.")

        @app.middleware("http")
        async def _guard_local_web(request: Request, call_next):  # noqa: ANN202
            try:
                host = urlsplit("http://" + request.headers.get("host", ""))
                valid_host = (
                    host.hostname in {"127.0.0.1", "localhost", "::1"}
                    and host.username is None and host.password is None
                    and not host.path and not host.query and not host.fragment
                )
                host.port
            except ValueError:
                valid_host = False
            origin = request.headers.get("origin")
            if (
                not valid_host
                or (origin is not None and origin not in web_origins)
                or request.headers.get("sec-fetch-site") == "cross-site"
            ):
                return JSONResponse(status_code=403, content=error_payload(
                    ErrorCode.UNAUTHORIZED, "This local service does not accept this request origin.",
                ))
            return await call_next(request)

    from ..secret_vault import SecretVaultError

    @app.exception_handler(SecretVaultError)
    async def _on_vault_error(_request: Request, _exc: SecretVaultError):  # noqa: ANN202
        return JSONResponse(status_code=503, content=error_payload(
            ErrorCode.PROVIDER_NOT_CONFIGURED,
            "Credential storage is locked or unavailable. Unlock it before changing settings.",
        ))

    if allowed_cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(allowed_cors_origins),
            allow_credentials=False,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
        )

    if mode == "token":

        @app.middleware("http")
        async def _require_bearer(request: Request, call_next):  # noqa: ANN202
            path = request.url.path
            # Let CORS preflight, the public health probe, and any non-API path through.
            if (
                request.method == "OPTIONS"
                or path in _PUBLIC_PATHS
                or not path.startswith(API_PREFIX)
            ):
                return await call_next(request)
            header = request.headers.get("authorization", "")
            provided = header[7:].strip() if header[:7].lower() == "bearer " else ""
            # Constant-time compare so a wrong token does not leak length/prefix via timing.
            if not provided or not secrets.compare_digest(provided, token):
                return JSONResponse(
                    status_code=401,
                    content=error_payload(
                        ErrorCode.UNAUTHORIZED,
                        "Missing or invalid bearer token.",
                    ),
                )
            return await call_next(request)

    @app.exception_handler(BaselineError)
    async def _on_baseline_error(_request: Request, exc: BaselineError):  # noqa: ANN202
        return JSONResponse(status_code=exc.http_status, content=exc.to_payload())

    @app.get(f"{API_PREFIX}/health")
    async def health() -> JsonDict:  # noqa: ANN202
        encryption_mode = "encrypted" if service.store.encrypted else "plaintext"
        return {
            "status": "ok",
            "fts5": service.store.fts5_available,
            "db_encrypted": service.store.encrypted,
            "database": {"encryption_mode": encryption_mode},
            "auth_mode": mode,
            "context_budget_preset": service.get_context_budget()["effective"]["preset"],
            "post_run_notes_modes": service.get_post_run_notes_config()["enabled_modes"],
            **service.vector_backend_status(),
            **service.job_backend_status(),
        }

    @app.get(f"{API_PREFIX}/providers")
    async def providers() -> JsonDict:  # noqa: ANN202
        return service.provider_status()

    @app.get(f"{API_PREFIX}/knowledge-graph/status")
    async def knowledge_graph_status(workspace_id: str = "default") -> JsonDict:  # noqa: ANN202
        return service.get_knowledge_graph_status(workspace_id=workspace_id)

    @app.get(f"{API_PREFIX}/knowledge-graph/jobs/{{job_id}}")
    async def knowledge_graph_job(  # noqa: ANN202
        job_id: str,
        workspace_id: str = "default",
    ) -> JsonDict:
        return service.get_knowledge_graph_rebuild_job(
            job_id,
            workspace_id=workspace_id,
        )

    @app.get(f"{API_PREFIX}/knowledge-graph/jobs/{{job_id}}/stream")
    async def knowledge_graph_job_stream(  # noqa: ANN202
        job_id: str,
        request: Request,
        workspace_id: str = "default",
        after_seq: str | None = None,
    ):
        cursor = _parse_job_event_cursor(
            after_seq,
            request.headers.get("Last-Event-ID"),
        )
        service.get_knowledge_graph_rebuild_job(
            job_id,
            workspace_id=workspace_id,
        )
        return StreamingResponse(
            service.stream_knowledge_graph_rebuild_job(
                job_id,
                workspace_id=workspace_id,
                after_seq=cursor,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get(f"{API_PREFIX}/knowledge-graph/graph")
    async def knowledge_graph_graph(  # noqa: ANN202
        workspace_id: str = "default",
        limit: int = 300,
        edge_types: str = "",
    ) -> JsonDict:
        parsed_edge_types = [item.strip() for item in edge_types.split(",") if item.strip()]
        return service.get_knowledge_graph(
            workspace_id=workspace_id,
            limit=limit,
            edge_types=parsed_edge_types,
        )

    @app.get(f"{API_PREFIX}/knowledge-graph/session-map")
    async def knowledge_graph_session_map(  # noqa: ANN202
        workspace_id: str = "default",
        session_limit: int = 50,
        member_preview: int = 0,
    ) -> JsonDict:
        return service.get_knowledge_graph_session_map(
            workspace_id=workspace_id,
            session_limit=session_limit,
            member_preview=member_preview,
        )

    @app.get(f"{API_PREFIX}/knowledge-graph/wiki-map")
    async def knowledge_graph_wiki_map(workspace_id: str = "default") -> JsonDict:  # noqa: ANN202
        return service.get_knowledge_graph_wiki_map(workspace_id=workspace_id)

    @app.post(f"{API_PREFIX}/knowledge-graph/wiki-map/refresh")
    async def knowledge_graph_wiki_map_refresh(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.start_knowledge_graph_wiki_refresh(body or {})

    @app.get(f"{API_PREFIX}/knowledge-graph/wiki-map/jobs/{{job_id}}")
    async def knowledge_graph_wiki_map_job(  # noqa: ANN202
        job_id: str,
        workspace_id: str = "default",
    ) -> JsonDict:
        return service.get_knowledge_graph_wiki_refresh_job(
            job_id,
            workspace_id=workspace_id,
        )

    @app.get(f"{API_PREFIX}/knowledge-graph/wiki-map/jobs/{{job_id}}/stream")
    async def knowledge_graph_wiki_map_job_stream(  # noqa: ANN202
        job_id: str,
        request: Request,
        workspace_id: str = "default",
        after_seq: str | None = None,
    ):
        cursor = _parse_job_event_cursor(
            after_seq,
            request.headers.get("Last-Event-ID"),
        )
        service.get_knowledge_graph_wiki_refresh_job(
            job_id,
            workspace_id=workspace_id,
        )
        return StreamingResponse(
            service.stream_knowledge_graph_wiki_refresh_job(
                job_id,
                workspace_id=workspace_id,
                after_seq=cursor,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get(f"{API_PREFIX}/knowledge-graph/nodes/{{node_id}}")
    async def knowledge_graph_node(node_id: str) -> JsonDict:  # noqa: ANN202
        return service.get_knowledge_graph_node(node_id)

    @app.post(f"{API_PREFIX}/knowledge-graph/rebuild")
    async def knowledge_graph_rebuild(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.start_knowledge_graph_rebuild(body or {})

    @app.put(f"{API_PREFIX}/providers")
    async def update_providers(body=Body(default=None)) -> JsonDict:  # noqa: ANN202, ANN001
        return service.update_providers(body)

    @app.get(f"{API_PREFIX}/providers/llm")
    async def llm_registry() -> JsonDict:  # noqa: ANN202
        return service.llm_registry_status()

    @app.put(f"{API_PREFIX}/providers/llm")
    async def upsert_llm_provider(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.upsert_llm_provider(body or {})

    @app.delete(f"{API_PREFIX}/providers/llm/{{provider_id}}")
    async def delete_llm_provider(provider_id: str) -> JsonDict:  # noqa: ANN202
        return service.remove_llm_provider(provider_id)

    @app.patch(f"{API_PREFIX}/providers/llm/{{provider_id}}")
    async def patch_llm_provider(provider_id: str, body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.patch_llm_provider(provider_id, body or {})

    @app.get(f"{API_PREFIX}/providers/llm/{{provider_id}}/models")
    async def list_provider_model_profiles(provider_id: str) -> JsonDict:  # noqa: ANN202
        return service.list_llm_provider_models(provider_id)

    @app.post(f"{API_PREFIX}/providers/llm/{{provider_id}}/models")
    async def create_provider_models(provider_id: str, body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.create_llm_provider_models(provider_id, body or {})

    @app.patch(f"{API_PREFIX}/providers/llm/{{provider_id}}/models/{{model_id}}")
    async def update_provider_model(provider_id: str, model_id: str, body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.update_llm_provider_model(provider_id, model_id, body or {})

    @app.delete(f"{API_PREFIX}/providers/llm/{{provider_id}}/models/{{model_id}}")
    async def delete_provider_model(provider_id: str, model_id: str) -> JsonDict:  # noqa: ANN202
        return service.delete_llm_provider_model(provider_id, model_id)

    @app.put(f"{API_PREFIX}/providers/roles")
    async def set_llm_roles(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.set_llm_roles(body or {})

    @app.post(f"{API_PREFIX}/providers/models")
    async def fetch_models(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.list_provider_models(body or {})

    @app.post(f"{API_PREFIX}/providers/test")
    async def test_llm(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.test_llm_provider(body or {})

    @app.post(f"{API_PREFIX}/tools/test")
    async def test_tools(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.test_search_tools(body or {})

    @app.post(f"{API_PREFIX}/settings/search-providers/save")
    async def save_search_providers(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.save_search_providers(body or {})

    @app.patch(f"{API_PREFIX}/settings/search-providers/enabled")
    async def toggle_search_provider(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.toggle_search_provider(body or {})

    @app.get(f"{API_PREFIX}/sessions")
    async def list_sessions() -> JsonDict:  # noqa: ANN202
        return {"sessions": service.list_sessions()}

    @app.get(f"{API_PREFIX}/sessions/archived")
    async def list_archived_sessions() -> JsonDict:  # noqa: ANN202
        return {"sessions": service.list_archived_sessions()}

    @app.post(f"{API_PREFIX}/sessions")
    async def create_session(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.create_session(title=str((body or {}).get("title") or ""))

    @app.patch(f"{API_PREFIX}/sessions/{{session_id}}")
    async def patch_session(session_id: str, body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.patch_session(session_id, body or {})

    @app.post(f"{API_PREFIX}/sessions/{{session_id}}/archive")
    async def archive_session(session_id: str) -> JsonDict:  # noqa: ANN202
        return service.archive_session(session_id)

    @app.post(f"{API_PREFIX}/sessions/{{session_id}}/restore")
    async def restore_session(session_id: str) -> JsonDict:  # noqa: ANN202
        return service.restore_session(session_id)

    @app.delete(f"{API_PREFIX}/sessions/{{session_id}}")
    async def delete_session(session_id: str) -> JsonDict:  # noqa: ANN202
        return service.delete_session(session_id)

    @app.get(f"{API_PREFIX}/sessions/{{session_id}}/messages")
    async def list_messages(session_id: str) -> JsonDict:  # noqa: ANN202
        return service.list_messages(session_id)

    @app.get(f"{API_PREFIX}/settings/context-budget")
    async def get_context_budget() -> JsonDict:  # noqa: ANN202
        return service.get_context_budget()

    @app.get(f"{API_PREFIX}/settings/search-usage")
    async def get_search_usage() -> JsonDict:  # noqa: ANN202
        return service.get_search_usage()

    @app.patch(f"{API_PREFIX}/settings/search-usage")
    async def update_search_usage(body=Body(default=None)) -> JsonDict:  # noqa: ANN202, ANN001
        return service.update_search_usage(body)

    @app.put(f"{API_PREFIX}/settings/context-budget")
    async def update_context_budget(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.update_context_budget(body or {})

    @app.get(f"{API_PREFIX}/settings/retrieval")
    async def get_retrieval_settings() -> JsonDict:  # noqa: ANN202
        return service.get_retrieval_settings()

    @app.patch(f"{API_PREFIX}/settings/retrieval")
    async def update_retrieval_settings(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.update_retrieval_settings(body or {})

    @app.get(f"{API_PREFIX}/settings/deep-checkpoint")
    async def get_deep_checkpoint_settings() -> JsonDict:  # noqa: ANN202
        return service.get_deep_checkpoint_settings()

    @app.patch(f"{API_PREFIX}/settings/deep-checkpoint")
    async def update_deep_checkpoint_settings(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.update_deep_checkpoint_settings(body or {})

    @app.get(f"{API_PREFIX}/settings/post-run-notes")
    async def get_post_run_notes() -> JsonDict:  # noqa: ANN202
        return service.get_post_run_notes_config()

    @app.put(f"{API_PREFIX}/settings/post-run-notes")
    async def update_post_run_notes(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.update_post_run_notes_config(body or {})

    @app.get(f"{API_PREFIX}/settings/memory")
    async def get_memory_settings() -> JsonDict:  # noqa: ANN202
        return service.get_memory_settings()

    @app.put(f"{API_PREFIX}/settings/memory")
    async def update_memory_settings(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.update_memory_settings(body or {})

    @app.post(f"{API_PREFIX}/settings/memory/format-probe")
    async def probe_memory_format() -> JsonDict:  # noqa: ANN202
        return service.probe_memory_format()

    @app.get(f"{API_PREFIX}/settings/knowledge-graph")
    async def get_knowledge_graph_settings() -> JsonDict:  # noqa: ANN202
        return service.get_knowledge_graph_config()

    @app.put(f"{API_PREFIX}/settings/knowledge-graph")
    async def update_knowledge_graph_settings(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.update_knowledge_graph_config(body or {})

    @app.delete(f"{API_PREFIX}/cache")
    async def purge_cache(scope: str = "all") -> JsonDict:  # noqa: ANN202
        return service.purge_cache(scope)

    @app.post(f"{API_PREFIX}/context-budget/recommend")
    async def recommend_context_budget(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.recommend_context_budget(body or {})

    @app.post(f"{API_PREFIX}/runs/estimate-cost")
    async def estimate_run_cost(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.estimate_run_cost(body or {})

    @app.post(f"{API_PREFIX}/runs")
    async def create_run(body: JsonDict) -> JsonDict:  # noqa: ANN202
        return service.create_run(body)

    @app.get(f"{API_PREFIX}/runs/{{run_id}}")
    async def get_run(run_id: str) -> JsonDict:  # noqa: ANN202
        return service.get_run(run_id)

    @app.post(f"{API_PREFIX}/runs/{{run_id}}/confidence/acknowledge")
    async def acknowledge_run_confidence(run_id: str) -> JsonDict:  # noqa: ANN202
        return service.acknowledge_run_confidence(run_id)

    @app.post(f"{API_PREFIX}/runs/{{run_id}}/regenerate")
    async def regenerate_run(run_id: str) -> JsonDict:  # noqa: ANN202
        return service.regenerate_run(run_id)

    @app.post(f"{API_PREFIX}/runs/{{run_id}}/cancel")
    async def cancel_run(run_id: str) -> JsonDict:  # noqa: ANN202
        return service.cancel_run(run_id)

    @app.post(f"{API_PREFIX}/runs/{{run_id}}/pause")
    async def pause_run(run_id: str) -> JsonDict:  # noqa: ANN202
        return service.pause_run(run_id)

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/checkpoint")
    async def get_run_checkpoint(run_id: str) -> JsonDict:  # noqa: ANN202
        return service.get_run_checkpoint(run_id)

    @app.post(f"{API_PREFIX}/runs/{{run_id}}/continue")
    async def continue_run(run_id: str, body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.continue_run_from_checkpoint(run_id, body or {})

    @app.post(f"{API_PREFIX}/runs/{{run_id}}/checkpoint/abandon")
    async def abandon_run_checkpoint(run_id: str, body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.abandon_run_checkpoint(run_id, body or {})

    @app.get(f"{API_PREFIX}/runs/{{run_id}}/stream")
    async def stream_run(run_id: str, request: Request, after_seq: int | None = None):  # noqa: ANN202
        raw_cursor: object = after_seq if after_seq is not None else request.headers.get("Last-Event-ID", "0")
        try:
            cursor = max(0, int(raw_cursor or 0))
        except (TypeError, ValueError) as error:
            raise BaselineError(ErrorCode.INVALID_REQUEST, "after_seq/Last-Event-ID must be a non-negative integer") from error
        return StreamingResponse(
            service.stream_run(run_id, after_seq=cursor),
            media_type="text/event-stream",
        )

    @app.get(f"{API_PREFIX}/memories")
    async def recall_memories(  # noqa: ANN202
        session_key: str = "",
        workspace_key: str = "default",
        user_key: str = "default",
        limit: int = 12,
        query: str = "",
        include_archived: bool = False,
    ) -> JsonDict:
        if include_archived:
            return {
                "memories": service.list_managed_memories(
                    {
                        "workspace_key": workspace_key,
                        "user_key": user_key,
                        "limit": limit,
                        "include_archived": True,
                    }
                )
            }
        items = service.recall_memory(
            {
                "session_key": session_key,
                "workspace_key": workspace_key,
                "user_key": user_key,
                "limit": limit,
                "query": query,
            }
        )
        return {"memories": items}

    @app.get(f"{API_PREFIX}/memories/archived")
    async def archived_memories(  # noqa: ANN202
        workspace_key: str = "default",
        user_key: str = "default",
        limit: int = 100,
    ) -> JsonDict:
        return service.list_archived_memories({"workspace_key": workspace_key, "user_key": user_key, "limit": limit})

    @app.get(f"{API_PREFIX}/memories/session")
    async def session_memories(  # noqa: ANN202
        workspace_key: str = "default",
        session_id: str = "",
        query: str = "",
        include_archived_sessions: bool = False,
        include_archived_memories: bool = False,
        limit: int = 50,
        offset: int = 0,
    ) -> JsonDict:
        return service.list_session_memories(
            {
                "workspace_key": workspace_key,
                "session_id": session_id,
                "query": query,
                "include_archived_sessions": include_archived_sessions,
                "include_archived_memories": include_archived_memories,
                "limit": limit,
                "offset": offset,
            }
        )

    @app.post(f"{API_PREFIX}/memories/batch-expiry")
    async def batch_memory_expiry(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.batch_memory_expiry(body or {})

    @app.post(f"{API_PREFIX}/memories/cleanup")
    async def cleanup_expired_memories(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.cleanup_expired_memories(body or {})

    @app.post(f"{API_PREFIX}/memories/optimize/preview")
    async def preview_memory_optimization(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.preview_memory_optimization(body or {})

    @app.post(f"{API_PREFIX}/memories/optimize/apply")
    async def apply_memory_optimization(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.apply_memory_optimization(body or {})

    @app.get(f"{API_PREFIX}/sessions/{{session_id}}/history/search")
    async def search_history(  # noqa: ANN202
        session_id: str, query: str = "", scope: str = "cross", limit: int = 10
    ) -> JsonDict:
        return service.search_history(session_id, {"query": query, "scope": scope, "limit": limit})

    @app.post(f"{API_PREFIX}/memories")
    async def add_memory(body: JsonDict) -> JsonDict:  # noqa: ANN202
        return service.add_memory(body)

    @app.get(f"{API_PREFIX}/memories/{{memory_id}}")
    async def get_memory(memory_id: str) -> JsonDict:  # noqa: ANN202
        return {"memory": service.get_memory(memory_id)}

    @app.post(f"{API_PREFIX}/memories/{{memory_id}}/archive")
    async def archive_memory(memory_id: str) -> JsonDict:  # noqa: ANN202
        return service.archive_memory(memory_id)

    @app.post(f"{API_PREFIX}/memories/{{memory_id}}/restore")
    async def restore_memory(memory_id: str) -> JsonDict:  # noqa: ANN202
        return service.restore_memory(memory_id)

    @app.patch(f"{API_PREFIX}/memories/{{memory_id}}")
    async def patch_memory(memory_id: str, body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.patch_memory(memory_id, body or {})

    @app.delete(f"{API_PREFIX}/memories/{{memory_id}}")
    async def delete_memory(memory_id: str) -> JsonDict:  # noqa: ANN202
        return service.delete_memory(memory_id)

    @app.get(f"{API_PREFIX}/memory/dream")
    async def memory_dream_status(workspace_key: str = "default", user_key: str = "default") -> JsonDict:  # noqa: ANN202
        return service.get_memory_dream_status({"workspace_key": workspace_key, "user_key": user_key})

    @app.get(f"{API_PREFIX}/memory/write-audit")
    async def memory_write_audit(workspace_key: str = "default", user_key: str = "default") -> JsonDict:  # noqa: ANN202
        return service.get_memory_write_audit({"workspace_key": workspace_key, "user_key": user_key})

    @app.post(f"{API_PREFIX}/memory/dream")
    async def start_memory_dream(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.start_memory_dream(body or {})

    @app.post(f"{API_PREFIX}/memory/dream/{{dream_run_id}}/cancel")
    async def cancel_memory_dream(dream_run_id: str) -> JsonDict:  # noqa: ANN202
        return service.cancel_memory_dream(dream_run_id)

    @app.put(f"{API_PREFIX}/memory/dream/schedule")
    async def update_memory_dream_schedule(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.update_memory_dream_schedule(body or {})

    @app.post(f"{API_PREFIX}/index/url")
    async def index_url(body: JsonDict) -> JsonDict:  # noqa: ANN202
        return service.index_url(body)

    @app.post(f"{API_PREFIX}/index/rebuild")
    async def rebuild_index(body: JsonDict | None = None) -> JsonDict:  # noqa: ANN202
        return service.rebuild_index(body or {})

    @app.get(f"{API_PREFIX}/index/documents")
    async def list_documents() -> JsonDict:  # noqa: ANN202
        return service.list_documents()

    @app.get(f"{API_PREFIX}/library/documents")
    async def list_library_documents() -> JsonDict:  # noqa: ANN202
        return service.list_library_documents()

    @app.post(f"{API_PREFIX}/library/documents/import")
    async def import_library_document(  # noqa: ANN202
        file: UploadFile = File(...),
        title: str = Form(""),
    ) -> JsonDict:
        try:
            content = await _read_upload_bounded(file, max_bytes=MAX_UPLOAD_BYTES)
        finally:
            await file.close()
        return await asyncio.to_thread(
            service.import_local_document,
            {
                "title": title,
                "filename": file.filename or "document",
                "mime_type": file.content_type or "",
                "content": content,
            },
        )

    @app.delete(f"{API_PREFIX}/library/documents/{{document_id}}")
    async def delete_library_document(document_id: str) -> JsonDict:  # noqa: ANN202
        return await asyncio.to_thread(service.delete_library_document, document_id)

    @app.get(f"{API_PREFIX}/library/notes")
    async def list_library_notes() -> JsonDict:  # noqa: ANN202
        return service.list_library_notes()

    @app.post(f"{API_PREFIX}/library/notes")
    async def add_library_note(body: JsonDict) -> JsonDict:  # noqa: ANN202
        return service.add_library_note(body)

    @app.get(f"{API_PREFIX}/library/notes/{{note_id}}/provenance")
    async def get_note_provenance(note_id: str) -> JsonDict:  # noqa: ANN202
        return service.get_note_provenance(note_id)

    @app.get(f"{API_PREFIX}/library/reports")
    async def list_library_reports() -> JsonDict:  # noqa: ANN202
        return service.list_library_reports()

    @app.post(f"{API_PREFIX}/library/reports")
    async def add_library_report(body: JsonDict) -> JsonDict:  # noqa: ANN202
        return service.add_report(body)

    @app.post(f"{API_PREFIX}/library/reports/upload")
    async def upload_library_report(  # noqa: ANN202
        title: str = Form(""),
        file: UploadFile = File(...),
    ) -> JsonDict:
        return service.upload_report(
            {
                "title": title,
                "filename": file.filename or "report.md",
                "content": await file.read(),
            }
        )

    @app.get(f"{API_PREFIX}/library/reports/{{report_id}}")
    async def get_library_report(report_id: str) -> JsonDict:  # noqa: ANN202
        return service.get_library_report(report_id)

    @app.get(f"{API_PREFIX}/library/sources/{{document_id}}/sessions")
    async def list_library_source_sessions(document_id: str) -> JsonDict:  # noqa: ANN202
        return service.list_source_sessions(document_id)

    @app.get(f"{API_PREFIX}/library/sources/{{document_id}}/snapshots")
    async def list_library_source_snapshots(document_id: str) -> JsonDict:  # noqa: ANN202
        return service.list_source_snapshots(document_id)

    @app.get(f"{API_PREFIX}/library/snapshots/{{snapshot_id}}")
    async def get_library_snapshot(snapshot_id: str) -> JsonDict:  # noqa: ANN202
        return service.get_library_snapshot(snapshot_id)

    @app.get(f"{API_PREFIX}/notes")
    async def list_notes() -> JsonDict:  # noqa: ANN202
        return service.list_notes()

    @app.post(f"{API_PREFIX}/notes")
    async def add_note(body: JsonDict) -> JsonDict:  # noqa: ANN202
        return service.add_note(body)

    @app.post(f"{API_PREFIX}/tools/enhance")
    async def enhance_prompt(body: JsonDict) -> JsonDict:  # noqa: ANN202
        return service.enhance_prompt(body)

    return app
