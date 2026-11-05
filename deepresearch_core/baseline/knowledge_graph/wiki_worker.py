"""RQ entrypoints for out-of-process Wiki Map refresh jobs."""

from __future__ import annotations

from ..worker import _get_service


def refresh_wiki_job(payload: dict, job_id: str) -> None:
    """RQ job: refresh the Wiki Map through the process-cached service."""

    service = _get_service()
    service._run_wiki_refresh_job_safe(payload or {}, job_id=job_id)  # noqa: SLF001 - keep RQ/thread job failure semantics aligned


def mark_wiki_refresh_failed(job, connection, exc_type=None, exc_value=None, traceback=None) -> None:  # noqa: ANN001
    """RQ failure callback: mark a killed/crashed Wiki refresh as failed."""

    try:
        service = _get_service()
        reason = getattr(exc_type, "__name__", None) or "unknown"
        message = "Wiki Map refresh terminated unexpectedly."
        if reason:
            message = f"{message} ({reason})"
        service.mark_knowledge_graph_wiki_refresh_failed(str(getattr(job, "id", "") or ""), message=message)
    except Exception:
        return
