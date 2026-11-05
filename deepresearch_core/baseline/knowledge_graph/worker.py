"""RQ entrypoints for out-of-process knowledge graph rebuild jobs."""

from __future__ import annotations

from ..worker import _get_service
from .store_ops import KnowledgeGraphJobInactive


class KnowledgeGraphWorkerError(RuntimeError):
    """Stable RQ boundary error that never carries provider or user content."""


def rebuild_job(payload: dict, job_id: str) -> None:
    """RQ job: rebuild the knowledge graph through the process-cached service."""

    service = _get_service()
    try:
        service.knowledge_graph_rebuild_now(payload or {}, job_id=job_id)
    except KnowledgeGraphJobInactive as error:
        if error.benign:
            # A terminal CAS or another valid worker already owns the job. Treat
            # this delivery as a no-op so RQ does not fail the authoritative job.
            return
        raise
    except Exception:
        # RQ persists exception text and formatted tracebacks in Redis. Do not let
        # an upstream prompt, local path, or secret cross that durable boundary.
        raise KnowledgeGraphWorkerError("Knowledge graph rebuild failed.") from None


def mark_rebuild_failed(job, connection, exc_type=None, exc_value=None, traceback=None) -> None:  # noqa: ANN001
    """RQ failure callback: mark a killed/crashed KG rebuild as failed."""

    try:
        service = _get_service()
        reason = getattr(exc_type, "__name__", None) or "unknown"
        message = "Knowledge graph rebuild terminated unexpectedly."
        if reason:
            message = f"{message} ({reason})"
        service.mark_knowledge_graph_rebuild_failed(str(getattr(job, "id", "") or ""), message=message)
    except Exception:
        return
