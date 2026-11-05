"""Durable, bounded progress events for Graph and Wiki maintenance jobs."""

from __future__ import annotations

import json
import math
import time
from typing import Any


JsonDict = dict[str, Any]

JOB_KINDS = frozenset({"graph", "wiki"})
TERMINAL_STATUSES = frozenset({"succeeded", "partial", "failed"})
MAX_EVENTS_PER_JOB = 1024
MAX_EVENT_PAYLOAD_BYTES = 16 * 1024

_PUBLIC_STAT_KEYS = {
    "graph": frozenset(
        {
            "sessions",
            "documents",
            "reports",
            "memories",
            "nodes",
            "edges",
            "provenance",
            "concept_nodes",
            "p2_edges",
            "concept_provenance",
            "concept_candidates",
            "concept_candidates_before_artifact_limit",
            "concept_candidates_truncated_by_artifact_limit",
            "concepts_before_workspace_limit",
            "concepts_truncated_by_workspace_limit",
            "p2_edges_before_workspace_limit",
            "p2_edges_truncated_by_workspace_limit",
            "concept_limit",
            "p2_edge_limit",
            "concepts_per_artifact_limit",
            "wiki_seed_count",
            "canonical_evidence_input_count",
        }
    ),
    "wiki": frozenset(
        {
            "attempted_sessions",
            "succeeded_sessions",
            "skipped_sessions",
            "failed_sessions",
            "eligible_sessions",
            "fresh_sessions",
            "carried_sessions",
            "uncovered_sessions",
            "candidate_coverage",
            "active_coverage",
            "orphan_sessions",
            "topic_count",
            "top_level_topic_count",
            "max_depth",
            "topics_with_evidence",
            "evidence_coverage",
            "question_like_title_count",
            "invalid_summary_count",
            "orphan_root_count",
            "topics",
            "tree_edges",
            "cross_links",
            "evidence_ref_repaired",
            "evidence_ref_dropped",
            "topic_dropped",
        }
    ),
}


def build_job_event_payload(job_kind: str, job: JsonDict) -> JsonDict:
    """Return the allowlisted, secret-safe public event payload for one job row."""

    kind = _normalize_kind(job_kind)
    status = str(job.get("status") or "").strip()
    payload: JsonDict = {
        "job_id": str(job.get("id") or ""),
        "job_kind": kind,
        "status": status,
        "stage": str(job.get("stage") or status or "unknown")[:80],
        "progress": _bounded_progress(job.get("progress")),
        "emitted_at": _event_time(job),
    }
    if kind == "wiki":
        for key in (
            "attempted_sessions",
            "succeeded_sessions",
            "skipped_sessions",
            "failed_sessions",
        ):
            payload[key] = _nonnegative_int(job.get(key))
    error_code = str(job.get("error_code") or "").strip()
    if error_code:
        payload["error_code"] = error_code[:80]
    safe_stats = _safe_stats(kind, job.get("stats_json"))
    if safe_stats:
        payload["stats"] = safe_stats
    encoded = _encode_payload(payload)
    if len(encoded.encode("utf-8")) > MAX_EVENT_PAYLOAD_BYTES:
        payload.pop("stats", None)
        encoded = _encode_payload(payload)
    if len(encoded.encode("utf-8")) > MAX_EVENT_PAYLOAD_BYTES:
        raise ValueError("knowledge graph job event payload exceeds 16 KiB")
    return payload


def append_job_event(
    conn: Any,
    *,
    job_kind: str,
    job: JsonDict,
    terminal: bool,
) -> int | None:
    """Append or deterministically coalesce one event inside the caller transaction."""

    kind = _normalize_kind(job_kind)
    job_id = str(job.get("id") or "").strip()
    if not job_id:
        raise ValueError("job event requires a job id")
    status = str(job.get("status") or "").strip()
    if terminal != (status in TERMINAL_STATUSES):
        raise ValueError("job event terminal flag does not match job status")

    payload = build_job_event_payload(kind, job)
    payload_json = _encode_payload(payload)
    created_at = str(payload["emitted_at"])
    last = conn.execute(
        "SELECT seq,event_type,payload_json FROM kg_job_events "
        "WHERE job_kind=? AND job_id=? ORDER BY seq DESC LIMIT 1",
        (kind, job_id),
    ).fetchone()
    count = int(
        conn.execute(
            "SELECT COUNT(*) FROM kg_job_events WHERE job_kind=? AND job_id=?",
            (kind, job_id),
        ).fetchone()[0]
    )
    last_seq = int(last["seq"]) if last is not None else 0
    event_type = "job_terminal" if terminal else "job_progress"

    if terminal:
        if last is not None and str(last["event_type"]) == "job_terminal":
            return None
        if count >= MAX_EVENTS_PER_JOB and last is not None:
            oldest_progress = conn.execute(
                "SELECT seq FROM kg_job_events "
                "WHERE job_kind=? AND job_id=? AND event_type='job_progress' "
                "ORDER BY seq ASC LIMIT 1",
                (kind, job_id),
            ).fetchone()
            if oldest_progress is not None:
                conn.execute(
                    "DELETE FROM kg_job_events WHERE job_kind=? AND job_id=? AND seq=?",
                    (kind, job_id, int(oldest_progress["seq"])),
                )
                count -= 1
        if count >= MAX_EVENTS_PER_JOB:
            raise ValueError("knowledge graph job event cap has no terminal slot")
        seq = last_seq + 1
        conn.execute(
            "INSERT INTO kg_job_events"
            "(job_kind,job_id,seq,event_type,payload_json,created_at) "
            "VALUES (?,?,?,?,?,?)",
            (kind, job_id, seq, event_type, payload_json, created_at),
        )
        return seq

    if last is not None and str(last["event_type"]) == "job_terminal":
        return None
    if last is not None and _same_progress_payload(last["payload_json"], payload):
        return None
    if count < MAX_EVENTS_PER_JOB - 1:
        seq = last_seq + 1
        conn.execute(
            "INSERT INTO kg_job_events"
            "(job_kind,job_id,seq,event_type,payload_json,created_at) "
            "VALUES (?,?,?,?,?,?)",
            (kind, job_id, seq, event_type, payload_json, created_at),
        )
        return seq
    return None


def list_job_events(
    conn: Any,
    *,
    job_kind: str,
    job_id: str,
    after_seq: int = 0,
) -> list[JsonDict]:
    """Return durable events after the authoritative cursor."""

    kind = _normalize_kind(job_kind)
    cursor = _nonnegative_int(after_seq)
    rows = conn.execute(
        "SELECT seq,event_type,payload_json,created_at FROM kg_job_events "
        "WHERE job_kind=? AND job_id=? AND seq>? ORDER BY seq ASC LIMIT ?",
        (kind, str(job_id or ""), cursor, MAX_EVENTS_PER_JOB),
    ).fetchall()
    events: list[JsonDict] = []
    for row in rows:
        try:
            payload = json.loads(str(row["payload_json"]))
        except (TypeError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        events.append(
            {
                "seq": int(row["seq"]),
                "event_type": str(row["event_type"]),
                "payload": payload,
                "created_at": float(row["created_at"]),
            }
        )
    return events


def synthetic_terminal_event(job_kind: str, job: JsonDict) -> JsonDict | None:
    """Build the one allowed legacy terminal frame without persisting it."""

    if str(job.get("status") or "") not in TERMINAL_STATUSES:
        return None
    return {
        "seq": 1,
        "event_type": "job_terminal",
        "payload": build_job_event_payload(job_kind, job),
        "created_at": _event_time(job),
        "synthetic": True,
    }


def _normalize_kind(value: str) -> str:
    kind = str(value or "").strip().lower()
    if kind not in JOB_KINDS:
        raise ValueError(f"unsupported knowledge graph job kind: {kind}")
    return kind


def _encode_payload(payload: JsonDict) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _bounded_progress(value: Any) -> float:
    try:
        progress = float(value or 0.0)
    except (TypeError, ValueError, OverflowError):
        return 0.0
    if not math.isfinite(progress):
        return 0.0
    return max(0.0, min(progress, 1.0))


def _nonnegative_int(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def _event_time(job: JsonDict) -> float:
    for key in ("finished_at", "updated_at", "started_at"):
        try:
            value = float(job.get(key) or 0.0)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(value) and value > 0:
            return value
    return time.time()


def _safe_stats(job_kind: str, value: Any) -> JsonDict:
    if not isinstance(value, dict):
        return {}
    safe: JsonDict = {}
    allowed = _PUBLIC_STAT_KEYS[_normalize_kind(job_kind)]
    for key in sorted(allowed):
        item = value.get(key)
        if isinstance(item, bool):
            safe[key] = item
        elif isinstance(item, int):
            safe[key] = item
        elif isinstance(item, float) and math.isfinite(item):
            safe[key] = item
    return safe


def _same_progress_payload(raw: Any, payload: JsonDict) -> bool:
    try:
        previous = json.loads(str(raw))
    except (TypeError, json.JSONDecodeError):
        return False
    if not isinstance(previous, dict):
        return False
    previous = {key: value for key, value in previous.items() if key != "emitted_at"}
    current = {key: value for key, value in payload.items() if key != "emitted_at"}
    return previous == current
