"""SQLite operations for Knowledge Graph Wiki Map snapshots and refresh jobs."""

from __future__ import annotations

import hashlib
import json
import re
import time
import uuid
from typing import Any, Iterable

from .job_events import append_job_event, list_job_events
from .wiki_models import JsonDict, WikiCrossLinkDraft, WikiSnapshotDraft, WikiTopicDraft, WikiTreeEdgeDraft


ACTIVE_STATUSES = {"queued", "running"}
TERMINAL_STATUSES = {"succeeded", "partial", "failed"}
STALE_QUEUED_JOB_SECONDS = 60.0
_SECRET_RE = re.compile(
    r"((?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{6,}|(?<![A-Za-z0-9])ghp_[A-Za-z0-9_]{6,}|AKIA[0-9A-Z]{8,}|"
    r"(?<![A-Za-z0-9])xox[abp]-[A-Za-z0-9-]{6,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)",
    re.IGNORECASE,
)


class WikiRefreshJobInactive(RuntimeError):
    """Raised when a worker loses ownership of a terminal Wiki refresh job."""


def _now() -> float:
    return time.time()


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _stable_id(prefix: str, *parts: str) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:20]}"


def _safe_json(raw: Any, fallback: Any = None) -> Any:
    if fallback is None:
        fallback = {}
    if raw in (None, ""):
        return fallback
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return fallback


def _safe_error_message(message: str) -> str:
    text = " ".join(str(message or "").split())
    if not text:
        return "Wiki Map refresh failed."
    if _SECRET_RE.search(text):
        return "Wiki Map refresh failed with a redacted error."
    return text[:240]


def _decode(row: Any | None) -> JsonDict | None:
    if row is None:
        return None
    data = dict(row)
    for key in ("stats_json", "attrs_json"):
        if key in data:
            data[key] = _safe_json(data.get(key), {})
    for key in ("session_ids_json", "source_message_ids_json", "evidence_refs_json"):
        if key in data:
            data[key] = _safe_json(data.get(key), [])
    for key in ("is_active", "stale"):
        if key in data:
            data[key] = bool(data[key])
    return data


class KnowledgeGraphWikiStore:
    """Bounded CRUD wrapper for versioned Wiki Map snapshots."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self._conn = store._conn
        self._lock = store._lock

    def create_refresh_job(
        self,
        *,
        workspace_id: str = "default",
        graph_version_id: str = "",
        backend: str = "thread",
    ) -> JsonDict:
        job_id = _uid("kgwikijob")
        ts = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO kg_wiki_refresh_jobs"
                "(id,snapshot_id,graph_version_id,workspace_id,backend,backend_job_id,status,stage,progress,"
                "started_at,finished_at,attempted_sessions,succeeded_sessions,skipped_sessions,failed_sessions,"
                "error_code,error_message,stats_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    job_id,
                    None,
                    graph_version_id or None,
                    workspace_id,
                    backend,
                    None,
                    "queued",
                    "queued",
                    0.0,
                    ts,
                    None,
                    0,
                    0,
                    0,
                    0,
                    None,
                    None,
                    "{}",
                ),
            )
            self._append_refresh_job_event(job_id, terminal=False)
            self._conn.commit()
        return self.get_refresh_job(job_id) or {"id": job_id, "status": "queued"}

    def set_refresh_job_backend_id(self, job_id: str, backend_job_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE kg_wiki_refresh_jobs SET backend_job_id=? "
                "WHERE id=? AND status NOT IN ('succeeded','partial','failed')",
                (backend_job_id, job_id),
            )
            self._conn.commit()

    def mark_refresh_job_running(
        self,
        job_id: str,
        *,
        snapshot_id: str,
        stage: str = "generating",
    ) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE kg_wiki_refresh_jobs SET snapshot_id=?, status='running', stage=?, progress=? "
                "WHERE id=? AND status NOT IN ('succeeded','partial','failed')",
                (snapshot_id, stage, 0.1, job_id),
            )
            if cursor.rowcount:
                self._append_refresh_job_event(job_id, terminal=False)
            self._conn.commit()
        return bool(cursor.rowcount)

    def update_refresh_job_progress(
        self,
        job_id: str,
        *,
        stage: str,
        attempted_sessions: int,
        succeeded_sessions: int,
        skipped_sessions: int,
        failed_sessions: int,
        progress: float,
        stats: JsonDict | None = None,
    ) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE kg_wiki_refresh_jobs "
                "SET stage=?, progress=?, attempted_sessions=?, succeeded_sessions=?, skipped_sessions=?, "
                "failed_sessions=?, stats_json=? "
                "WHERE id=? AND status NOT IN ('succeeded','partial','failed')",
                (
                    stage,
                    max(0.0, min(float(progress), 0.99)),
                    int(attempted_sessions),
                    int(succeeded_sessions),
                    int(skipped_sessions),
                    int(failed_sessions),
                    json.dumps(stats or {}),
                    job_id,
                ),
            )
            if cursor.rowcount:
                self._append_refresh_job_event(job_id, terminal=False)
            self._conn.commit()
        return bool(cursor.rowcount)

    def mark_refresh_job_succeeded(
        self,
        job_id: str,
        *,
        status: str = "succeeded",
        stats: JsonDict | None = None,
    ) -> bool:
        if status not in {"succeeded", "partial"}:
            raise ValueError("Wiki Map terminal success status must be succeeded or partial")
        stage = "completed" if status == "succeeded" else "partial"
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE kg_wiki_refresh_jobs "
                "SET status=?, stage=?, progress=1.0, finished_at=?, stats_json=? "
                "WHERE id=? AND status NOT IN ('succeeded','partial','failed')",
                (status, stage, _now(), json.dumps(stats or {}), job_id),
            )
            if cursor.rowcount:
                self._append_refresh_job_event(job_id, terminal=True)
            self._conn.commit()
        return bool(cursor.rowcount)

    def mark_refresh_job_failed(
        self,
        job_id: str,
        *,
        error_code: str = "INTERNAL_ERROR",
        error_message: str = "Wiki Map refresh failed.",
        stats: JsonDict | None = None,
    ) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE kg_wiki_refresh_jobs "
                "SET status='failed', stage='failed', progress=1.0, finished_at=?, "
                "error_code=?, error_message=?, stats_json=? "
                "WHERE id=? AND status NOT IN ('succeeded','partial','failed')",
                (_now(), error_code, _safe_error_message(error_message), json.dumps(stats or {}), job_id),
            )
            if cursor.rowcount:
                self._append_refresh_job_event(job_id, terminal=True)
            self._conn.commit()
        return bool(cursor.rowcount)

    def get_refresh_job(self, job_id: str) -> JsonDict | None:
        row = self._conn.execute("SELECT * FROM kg_wiki_refresh_jobs WHERE id=?", (job_id,)).fetchone()
        return _public_job(row)

    def list_refresh_job_events(
        self,
        job_id: str,
        *,
        after_seq: int = 0,
    ) -> list[JsonDict]:
        with self._lock:
            return list_job_events(
                self._conn,
                job_kind="wiki",
                job_id=job_id,
                after_seq=after_seq,
            )

    def get_running_refresh_job(self, workspace_id: str = "default") -> JsonDict | None:
        placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
        rows = self._conn.execute(
            f"SELECT * FROM kg_wiki_refresh_jobs WHERE workspace_id=? AND status IN ({placeholders}) "
            "ORDER BY started_at DESC LIMIT 1",
            (workspace_id, *sorted(ACTIVE_STATUSES)),
        ).fetchall()
        return _public_job(rows[0]) if rows else None

    def get_latest_refresh_job(self, workspace_id: str = "default") -> JsonDict | None:
        row = self._conn.execute(
            "SELECT * FROM kg_wiki_refresh_jobs WHERE workspace_id=? ORDER BY started_at DESC LIMIT 1",
            (workspace_id,),
        ).fetchone()
        return _public_job(row)

    def fail_stale_orphan_refresh_jobs(
        self,
        *,
        workspace_id: str = "default",
        older_than_s: float = STALE_QUEUED_JOB_SECONDS,
    ) -> int:
        cutoff = _now() - max(float(older_than_s), 0.0)
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM kg_wiki_refresh_jobs "
                "WHERE workspace_id=? AND status='queued' "
                "AND backend_job_id IS NULL AND started_at<=? ORDER BY id",
                (workspace_id, cutoff),
            ).fetchall()
            changed = 0
            for row in rows:
                job_id = str(row["id"])
                cursor = self._conn.execute(
                    "UPDATE kg_wiki_refresh_jobs "
                    "SET status='failed', stage='failed', progress=1.0, finished_at=?, "
                    "error_code='ENQUEUE_FAILED', "
                    "error_message='Wiki Map refresh could not be queued. Please retry.' "
                    "WHERE id=? AND status='queued'",
                    (_now(), job_id),
                )
                if cursor.rowcount:
                    changed += 1
                    self._append_refresh_job_event(job_id, terminal=True)
            self._conn.commit()
        return changed

    def _append_refresh_job_event(self, job_id: str, *, terminal: bool) -> int | None:
        row = self._conn.execute(
            "SELECT * FROM kg_wiki_refresh_jobs WHERE id=?",
            (job_id,),
        ).fetchone()
        job = _public_job(row)
        if job is None:
            return None
        job["stats_json"] = dict(job.get("stats") or {})
        return append_job_event(
            self._conn,
            job_kind="wiki",
            job=job,
            terminal=terminal,
        )

    def create_staging_snapshot(self, *, workspace_id: str = "default", graph_version_id: str = "") -> str:
        snapshot_id = _uid("kgwiki")
        ts = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO kg_wiki_snapshots"
                "(id,workspace_id,graph_version_id,status,is_active,stale,stale_reason,started_at,finished_at,"
                "activated_at,stats_json,error_code,error_message) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    snapshot_id,
                    workspace_id,
                    graph_version_id or None,
                    "staging",
                    0,
                    0,
                    None,
                    ts,
                    None,
                    None,
                    "{}",
                    None,
                    None,
                ),
            )
            self._conn.commit()
        return snapshot_id

    def begin_refresh_job_snapshot(
        self,
        job_id: str,
        *,
        workspace_id: str = "default",
        graph_version_id: str = "",
        stage: str = "collecting_sessions",
    ) -> str | None:
        """Atomically claim a queued refresh job and create its staging snapshot."""

        snapshot_id = _uid("kgwiki")
        ts = _now()
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO kg_wiki_snapshots"
                    "(id,workspace_id,graph_version_id,status,is_active,stale,stale_reason,started_at,finished_at,"
                    "activated_at,stats_json,error_code,error_message) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        snapshot_id,
                        workspace_id,
                        graph_version_id or None,
                        "staging",
                        0,
                        0,
                        None,
                        ts,
                        None,
                        None,
                        "{}",
                        None,
                        None,
                    ),
                )
                cursor = self._conn.execute(
                    "UPDATE kg_wiki_refresh_jobs "
                    "SET snapshot_id=?, graph_version_id=?, status='running', stage=?, progress=? "
                    "WHERE id=? AND workspace_id=? AND status='queued'",
                    (snapshot_id, graph_version_id or None, stage, 0.1, job_id, workspace_id),
                )
                if not cursor.rowcount:
                    self._conn.rollback()
                    return None
                self._append_refresh_job_event(job_id, terminal=False)
                self._conn.commit()
                return snapshot_id
            except Exception:
                self._conn.rollback()
                raise

    def replace_snapshot_contents(
        self,
        snapshot_id: str,
        draft: WikiSnapshotDraft,
        *,
        job_id: str | None = None,
    ) -> JsonDict:
        topic_rows: list[tuple[Any, ...]] = []
        topic_ids: dict[str, str] = {}
        for topic in sorted(draft.topics, key=lambda item: (item.depth, item.topic_key)):
            topic_id = _stable_id("kgwikit", snapshot_id, topic.topic_key)
            topic_ids[topic.topic_key] = topic_id
            topic_rows.append(_topic_row(snapshot_id, topic, topic_id))

        tree_rows: list[tuple[Any, ...]] = []
        for edge in sorted(draft.tree_edges, key=lambda item: (item.src_key, item.dst_key, item.edge_type)):
            src_id = topic_ids.get(edge.src_key)
            dst_id = topic_ids.get(edge.dst_key)
            if not src_id or not dst_id:
                continue
            tree_rows.append(_tree_edge_row(snapshot_id, edge, src_id, dst_id))

        cross_rows: list[tuple[Any, ...]] = []
        for link in sorted(draft.cross_links, key=lambda item: (item.src_key, item.dst_key, item.evidence_type)):
            src_id = topic_ids.get(link.src_key)
            dst_id = topic_ids.get(link.dst_key)
            if not src_id or not dst_id:
                continue
            cross_rows.append(_cross_link_row(snapshot_id, link, src_id, dst_id))

        stats = {
            "topics": len(topic_rows),
            "tree_edges": len(tree_rows),
            "cross_links": len(cross_rows),
            **dict(draft.stats or {}),
        }
        with self._lock:
            try:
                if job_id:
                    cursor = self._conn.execute(
                        "UPDATE kg_wiki_refresh_jobs SET stage=stage "
                        "WHERE id=? AND snapshot_id=? AND status='running'",
                        (job_id, snapshot_id),
                    )
                    if not cursor.rowcount:
                        self._conn.rollback()
                        raise WikiRefreshJobInactive(job_id)
                self._conn.execute("DELETE FROM kg_wiki_cross_links WHERE snapshot_id=?", (snapshot_id,))
                self._conn.execute("DELETE FROM kg_wiki_tree_edges WHERE snapshot_id=?", (snapshot_id,))
                self._conn.execute("DELETE FROM kg_wiki_topics WHERE snapshot_id=?", (snapshot_id,))
                self._conn.executemany(
                    "INSERT INTO kg_wiki_topics"
                    "(id,snapshot_id,topic_key,title,summary,depth,parent_id,session_ids_json,"
                    "source_message_ids_json,attrs_json,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    topic_rows,
                )
                self._conn.executemany(
                    "INSERT INTO kg_wiki_tree_edges"
                    "(id,snapshot_id,src_topic_id,dst_topic_id,edge_type,attrs_json,created_at) "
                    "VALUES (?,?,?,?,?,?,?)",
                    tree_rows,
                )
                self._conn.executemany(
                    "INSERT INTO kg_wiki_cross_links"
                    "(id,snapshot_id,src_topic_id,dst_topic_id,evidence_type,evidence_refs_json,attrs_json,created_at) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    cross_rows,
                )
                self._conn.execute(
                    "UPDATE kg_wiki_snapshots SET stats_json=? WHERE id=?",
                    (json.dumps(stats), snapshot_id),
                )
                self._conn.commit()
                return stats
            except Exception:
                self._conn.rollback()
                raise

    def activate_snapshot(
        self,
        snapshot_id: str,
        *,
        status: str = "succeeded",
        stats: JsonDict | None = None,
        job_id: str | None = None,
    ) -> None:
        if status not in {"succeeded", "partial"}:
            raise ValueError("Wiki Map terminal success status must be succeeded or partial")
        ts = _now()
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT workspace_id FROM kg_wiki_snapshots WHERE id=?", (snapshot_id,)
                ).fetchone()
                if not row:
                    raise ValueError(f"unknown wiki snapshot: {snapshot_id}")
                workspace_id = str(row["workspace_id"])
                if job_id:
                    stage = "completed" if status == "succeeded" else "partial"
                    cursor = self._conn.execute(
                        "UPDATE kg_wiki_refresh_jobs "
                        "SET status=?, stage=?, progress=1.0, finished_at=?, stats_json=? "
                        "WHERE id=? AND snapshot_id=? AND status='running'",
                        (status, stage, ts, json.dumps(stats or {}), job_id, snapshot_id),
                    )
                    if not cursor.rowcount:
                        self._conn.rollback()
                        raise WikiRefreshJobInactive(job_id)
                self._conn.execute(
                    "UPDATE kg_wiki_snapshots SET is_active=0 WHERE workspace_id=? AND is_active=1",
                    (workspace_id,),
                )
                self._conn.execute(
                    "UPDATE kg_wiki_snapshots "
                    "SET status=?, is_active=1, stale=0, stale_reason=NULL, finished_at=?, activated_at=?, stats_json=? "
                    "WHERE id=?",
                    (status, ts, ts, json.dumps(stats or {}), snapshot_id),
                )
                if job_id:
                    self._append_refresh_job_event(job_id, terminal=True)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def mark_snapshot_failed(
        self,
        snapshot_id: str,
        *,
        error_code: str = "INTERNAL_ERROR",
        error_message: str = "Wiki Map refresh failed.",
        stats: JsonDict | None = None,
        job_id: str | None = None,
    ) -> bool:
        with self._lock:
            if job_id:
                cursor = self._conn.execute(
                    "UPDATE kg_wiki_refresh_jobs SET stage=stage "
                    "WHERE id=? AND snapshot_id=? AND status='running'",
                    (job_id, snapshot_id),
                )
                if not cursor.rowcount:
                    self._conn.rollback()
                    return False
            cursor = self._conn.execute(
                "UPDATE kg_wiki_snapshots "
                "SET status='failed', is_active=0, finished_at=?, stats_json=?, error_code=?, error_message=? "
                "WHERE id=?",
                (_now(), json.dumps(stats or {}), error_code, _safe_error_message(error_message), snapshot_id),
            )
            self._conn.commit()
        return bool(cursor.rowcount)

    def mark_active_stale(self, workspace_id: str = "default", *, reason: str = "graph_version_changed") -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE kg_wiki_snapshots SET stale=1, stale_reason=? WHERE workspace_id=? AND is_active=1",
                (reason, workspace_id),
            )
            self._conn.commit()

    def get_active_snapshot(self, workspace_id: str = "default") -> JsonDict | None:
        row = self._conn.execute(
            "SELECT * FROM kg_wiki_snapshots WHERE workspace_id=? AND is_active=1 LIMIT 1",
            (workspace_id,),
        ).fetchone()
        return _public_snapshot(row)

    def get_snapshot_topics(self, snapshot_id: str) -> list[JsonDict]:
        rows = self._conn.execute(
            "SELECT * FROM kg_wiki_topics WHERE snapshot_id=? ORDER BY depth ASC, title ASC, topic_key ASC",
            (snapshot_id,),
        ).fetchall()
        return [_public_topic(row) for row in rows]

    def get_snapshot_tree_edges(self, snapshot_id: str) -> list[JsonDict]:
        rows = self._conn.execute(
            "SELECT * FROM kg_wiki_tree_edges WHERE snapshot_id=? ORDER BY edge_type ASC, src_topic_id ASC, dst_topic_id ASC",
            (snapshot_id,),
        ).fetchall()
        return [_public_tree_edge(row) for row in rows]

    def get_snapshot_cross_links(self, snapshot_id: str) -> list[JsonDict]:
        rows = self._conn.execute(
            "SELECT * FROM kg_wiki_cross_links WHERE snapshot_id=? ORDER BY evidence_type ASC, src_topic_id ASC, dst_topic_id ASC",
            (snapshot_id,),
        ).fetchall()
        return [_public_cross_link(row) for row in rows]


def _topic_row(snapshot_id: str, topic: WikiTopicDraft, topic_id: str) -> tuple[Any, ...]:
    return (
        topic_id,
        snapshot_id,
        topic.topic_key,
        topic.title,
        topic.summary,
        int(topic.depth),
        _stable_id("kgwikit", snapshot_id, topic.parent_key) if topic.parent_key else None,
        json.dumps(list(topic.session_ids)),
        json.dumps(list(topic.source_message_ids)),
        json.dumps(topic.attrs or {}),
        _now(),
    )


def _tree_edge_row(snapshot_id: str, edge: WikiTreeEdgeDraft, src_id: str, dst_id: str) -> tuple[Any, ...]:
    return (
        _stable_id("kgwikie", snapshot_id, edge.edge_type, src_id, dst_id),
        snapshot_id,
        src_id,
        dst_id,
        edge.edge_type,
        json.dumps(edge.attrs or {}),
        _now(),
    )


def _cross_link_row(snapshot_id: str, link: WikiCrossLinkDraft, src_id: str, dst_id: str) -> tuple[Any, ...]:
    return (
        _stable_id("kgwikic", snapshot_id, link.evidence_type, src_id, dst_id),
        snapshot_id,
        src_id,
        dst_id,
        link.evidence_type,
        json.dumps(list(link.evidence_refs)),
        json.dumps(link.attrs or {}),
        _now(),
    )


def _public_snapshot(row: Any | None) -> JsonDict | None:
    data = _decode(row)
    if not data:
        return None
    return {
        "id": data.get("id"),
        "workspace_id": data.get("workspace_id"),
        "graph_version_id": data.get("graph_version_id") or "",
        "status": data.get("status"),
        "is_active": bool(data.get("is_active")),
        "stale": bool(data.get("stale")),
        "stale_reason": data.get("stale_reason") or "",
        "started_at": data.get("started_at"),
        "finished_at": data.get("finished_at"),
        "activated_at": data.get("activated_at"),
        "stats": data.get("stats_json") or {},
        "error_code": data.get("error_code") or "",
        "error_message": data.get("error_message") or "",
    }


def _public_topic(row: Any | None) -> JsonDict:
    data = _decode(row) or {}
    return {
        "id": data.get("id"),
        "snapshot_id": data.get("snapshot_id"),
        "key": data.get("topic_key"),
        "title": data.get("title") or "",
        "summary": data.get("summary") or "",
        "depth": int(data.get("depth") or 0),
        "parent_id": data.get("parent_id") or "",
        "session_ids": data.get("session_ids_json") or [],
        "source_message_ids": data.get("source_message_ids_json") or [],
        "attrs": data.get("attrs_json") or {},
    }


def _public_tree_edge(row: Any | None) -> JsonDict:
    data = _decode(row) or {}
    return {
        "id": data.get("id"),
        "snapshot_id": data.get("snapshot_id"),
        "source": data.get("src_topic_id"),
        "target": data.get("dst_topic_id"),
        "type": data.get("edge_type") or "parent_child",
        "attrs": data.get("attrs_json") or {},
    }


def _public_cross_link(row: Any | None) -> JsonDict:
    data = _decode(row) or {}
    return {
        "id": data.get("id"),
        "snapshot_id": data.get("snapshot_id"),
        "source": data.get("src_topic_id"),
        "target": data.get("dst_topic_id"),
        "type": data.get("evidence_type") or "",
        "evidence_type": data.get("evidence_type") or "",
        "evidence_refs": data.get("evidence_refs_json") or [],
        "attrs": data.get("attrs_json") or {},
    }


def _public_job(row: Any | None) -> JsonDict | None:
    data = _decode(row)
    if not data:
        return None
    return {
        "id": data.get("id"),
        "snapshot_id": data.get("snapshot_id") or "",
        "graph_version_id": data.get("graph_version_id") or "",
        "workspace_id": data.get("workspace_id") or "default",
        "backend": data.get("backend") or "",
        "backend_job_id": data.get("backend_job_id") or "",
        "status": data.get("status") or "",
        "stage": data.get("stage") or "",
        "progress": data.get("progress") or 0.0,
        "started_at": data.get("started_at"),
        "finished_at": data.get("finished_at"),
        "attempted_sessions": int(data.get("attempted_sessions") or 0),
        "succeeded_sessions": int(data.get("succeeded_sessions") or 0),
        "skipped_sessions": int(data.get("skipped_sessions") or 0),
        "failed_sessions": int(data.get("failed_sessions") or 0),
        "error_code": data.get("error_code") or "",
        "error_message": data.get("error_message") or "",
        "stats": data.get("stats_json") or {},
    }
