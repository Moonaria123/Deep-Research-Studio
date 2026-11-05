"""Observe-only Knowledge Graph status and freshness helpers."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from .read_support import public_job, public_version


JsonDict = dict[str, Any]


def read_knowledge_graph_status(
    *,
    ops: Any,
    builder: Any,
    connection: Any,
    workspace_id: str,
    event_limit: int,
    deadline: float,
    clock: Callable[[], float],
) -> JsonDict:
    """Return bounded current status without triggering a projection rebuild."""

    bounded_events = max(1, min(int(event_limit), 20))
    status = ops.get_active_status(workspace_id)
    active = status.get("active_version")
    active = active if isinstance(active, dict) else None
    stored = active.get("base_snapshot_json") if active else None
    stored = stored if isinstance(stored, dict) else {}
    stored_signature = str(stored.get("canonical_evidence_signature") or "")
    current_signature = ""
    freshness = "unknown"
    reason_codes: list[str] = []
    try:
        _check(deadline, clock)
        connection.set_progress_handler(lambda: int(clock() > deadline), 1_000)
        try:
            current = builder.current_base_snapshot(workspace_id=workspace_id)
        finally:
            connection.set_progress_handler(None, 0)
        _check(deadline, clock)
        current_signature = str(current.get("canonical_evidence_signature") or "")
    except TimeoutError:
        reason_codes.append("kg_read_budget_exceeded")
    except Exception:  # noqa: BLE001 - status remains observe-only and secret-safe
        reason_codes.append(
            "kg_read_budget_exceeded" if clock() > deadline else "kg_current_signature_unavailable"
        )
    if stored_signature and current_signature:
        freshness = "fresh" if stored_signature == current_signature else "stale"
        if freshness == "stale":
            reason_codes.append("kg_snapshot_signature_mismatch")
    elif active is None:
        reason_codes.append("kg_active_version_unavailable")
    else:
        reason_codes.append("kg_snapshot_signature_unavailable")

    job = status.get("last_build_job")
    job = job if isinstance(job, dict) else None
    events: list[JsonDict] = []
    if job and clock() <= deadline:
        events = _recent_job_events(connection, str(job.get("id") or ""), bounded_events)
        if clock() > deadline:
            events = []
            freshness = "unknown"
            if "kg_read_budget_exceeded" not in reason_codes:
                reason_codes.append("kg_read_budget_exceeded")
    elif job and "kg_read_budget_exceeded" not in reason_codes:
        reason_codes.append("kg_read_budget_exceeded")
    return {
        "active_version": public_version(active) if active else None,
        "counts": {
            "nodes": int((status.get("counts") or {}).get("nodes") or 0),
            "edges": int((status.get("counts") or {}).get("edges") or 0),
        },
        "stored_signature": stored_signature,
        "current_signature": current_signature,
        "freshness": freshness,
        "last_build_job": public_job(job) if job else None,
        "recent_events": events,
        "truncated": False,
        "reason_codes": list(dict.fromkeys(reason_codes)),
    }


def _recent_job_events(connection: Any, job_id: str, limit: int) -> list[JsonDict]:
    rows = connection.execute(
        "SELECT seq,event_type,payload_json,created_at FROM kg_job_events "
        "WHERE job_kind='graph' AND job_id=? ORDER BY seq DESC LIMIT ?",
        (job_id, limit),
    ).fetchall()
    events: list[JsonDict] = []
    for row in reversed(rows):
        try:
            payload = json.loads(str(row["payload_json"] or "{}"))
        except json.JSONDecodeError:
            payload = {}
        events.append(
            {
                "seq": int(row["seq"]),
                "event_type": str(row["event_type"] or ""),
                "payload": payload if isinstance(payload, dict) else {},
                "created_at": float(row["created_at"] or 0.0),
            }
        )
    return events


def _check(deadline: float, clock: Callable[[], float]) -> None:
    if clock() > deadline:
        raise TimeoutError("knowledge graph read budget exceeded")
