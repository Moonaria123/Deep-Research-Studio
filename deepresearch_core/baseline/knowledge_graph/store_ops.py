"""SQLite operations for versioned knowledge graph projections."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from typing import Any, Iterable

from .job_events import append_job_event, list_job_events
from .models import EdgeDraft, JsonDict, NodeDraft, ProvenanceDraft


ACTIVE_STATUSES = {"queued", "running"}
TERMINAL_STATUSES = {"succeeded", "partial", "failed"}
STALE_QUEUED_JOB_SECONDS = 60.0


class KnowledgeGraphJobInactive(RuntimeError):
    """Raised when a worker cannot safely retain ownership of a build job."""

    def __init__(
        self,
        job_id: str,
        *,
        reason: str = "inactive",
        status: str | None = None,
    ) -> None:
        self.job_id = str(job_id or "")
        self.reason = str(reason or "inactive")
        self.status = str(status or "") or None
        self.terminal = self.status in TERMINAL_STATUSES
        self.benign = self.terminal or self.reason in {
            "already_running",
            "version_mismatch",
        }
        super().__init__(f"knowledge graph job inactive: {self.reason}")


def _now() -> float:
    return time.time()


def _uid(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _stable_id(prefix: str, *parts: str) -> str:
    raw = "\x1f".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:20]}"


def _safe_json(raw: Any) -> Any:
    if raw in (None, ""):
        return {}
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return {}


def _decode(row: Any | None) -> JsonDict | None:
    if row is None:
        return None
    data = dict(row)
    for key in ("base_snapshot_json", "stats_json", "attrs_json", "span_json", "evidence_json"):
        if key in data:
            data[key] = _safe_json(data.get(key))
    if "is_active" in data:
        data["is_active"] = bool(data["is_active"])
    if "visible" in data:
        data["visible"] = bool(data["visible"])
    return data


class KnowledgeGraphStore:
    """Bounded CRUD wrapper over the host Store's SQLite connection."""

    def __init__(self, store: Any) -> None:
        self._store = store
        self._conn = store._conn
        self._lock = store._lock

    def _inactive_job_error(
        self,
        job_id: str,
        *,
        workspace_id: str | None = None,
        version_id: str | None = None,
    ) -> KnowledgeGraphJobInactive:
        row = self._conn.execute(
            "SELECT workspace_id,version_id,status FROM kg_build_jobs WHERE id=?",
            (job_id,),
        ).fetchone()
        if row is None:
            return KnowledgeGraphJobInactive(job_id, reason="not_found")
        status = str(row["status"] or "")
        if status in TERMINAL_STATUSES:
            return KnowledgeGraphJobInactive(
                job_id,
                reason="terminal",
                status=status,
            )
        if workspace_id is not None and str(row["workspace_id"] or "") != workspace_id:
            reason = "workspace_mismatch"
        elif version_id is not None and str(row["version_id"] or "") != version_id:
            reason = "version_mismatch"
        elif status == "running":
            reason = "already_running"
        else:
            reason = f"status_{status or 'unknown'}"
        return KnowledgeGraphJobInactive(job_id, reason=reason, status=status)

    def create_staging_version(self, *, workspace_id: str = "default", base_snapshot: JsonDict | None = None) -> str:
        version_id = _uid("kgv")
        ts = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO kg_graph_versions"
                "(id,workspace_id,status,is_active,started_at,finished_at,activated_at,base_snapshot_json,stats_json) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (version_id, workspace_id, "staging", 0, ts, None, None, json.dumps(base_snapshot or {}), "{}"),
            )
            self._conn.commit()
        return version_id

    def begin_build_job_version(
        self,
        job_id: str,
        *,
        workspace_id: str = "default",
        base_snapshot: JsonDict | None = None,
        stage: str = "projecting",
    ) -> str:
        """Atomically claim a queued job and create its staging version."""

        version_id = _uid("kgv")
        ts = _now()
        with self._lock:
            try:
                self._conn.execute(
                    "INSERT INTO kg_graph_versions"
                    "(id,workspace_id,status,is_active,started_at,finished_at,activated_at,base_snapshot_json,stats_json) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        version_id,
                        workspace_id,
                        "staging",
                        0,
                        ts,
                        None,
                        None,
                        json.dumps(base_snapshot or {}),
                        "{}",
                    ),
                )
                cursor = self._conn.execute(
                    "UPDATE kg_build_jobs SET version_id=?, status='running', stage=?, progress=? "
                    "WHERE id=? AND workspace_id=? AND status='queued'",
                    (version_id, stage, 0.1, job_id, workspace_id),
                )
                if not cursor.rowcount:
                    self._conn.rollback()
                    inactive = self._inactive_job_error(
                        job_id,
                        workspace_id=workspace_id,
                    )
                    if inactive.reason == "workspace_mismatch":
                        failed = self._conn.execute(
                            "UPDATE kg_build_jobs "
                            "SET status='failed', stage='failed', progress=1.0, finished_at=?, "
                            "error_code='JOB_CLAIM_REJECTED', "
                            "error_message='Knowledge graph rebuild could not safely claim its queued job.' "
                            "WHERE id=? AND status IN ('queued','running')",
                            (_now(), job_id),
                        )
                        if failed.rowcount:
                            self._append_build_job_event(job_id, terminal=True)
                        self._conn.commit()
                    raise inactive
                self._append_build_job_event(job_id, terminal=False)
                self._conn.commit()
                return version_id
            except Exception:
                self._conn.rollback()
                raise

    def replace_version_contents(
        self,
        version_id: str,
        *,
        nodes: Iterable[NodeDraft],
        edges: Iterable[EdgeDraft],
        provenance: Iterable[ProvenanceDraft],
        base_snapshot: JsonDict | None = None,
        job_id: str | None = None,
    ) -> JsonDict:
        node_rows: list[tuple[Any, ...]] = []
        node_ids: dict[str, str] = {}
        for node in sorted(nodes, key=lambda item: (item.node_type, item.node_key)):
            node_id = _stable_id("kgn", version_id, node.node_key)
            node_ids[node.node_key] = node_id
            node_rows.append(
                (
                    node_id,
                    version_id,
                    node.node_type,
                    node.node_key,
                    node.title,
                    node.summary,
                    json.dumps(node.attrs),
                    float(node.confidence),
                    int(bool(node.visible)),
                    _now(),
                )
            )

        edge_rows: list[tuple[Any, ...]] = []
        edge_ids: dict[str, str] = {}
        for edge in sorted(edges, key=lambda item: (item.edge_type, item.src_key, item.dst_key)):
            src_id = node_ids.get(edge.src_key)
            dst_id = node_ids.get(edge.dst_key)
            if not src_id or not dst_id:
                continue
            edge_id = _stable_id("kge", version_id, edge.edge_type, src_id, dst_id)
            edge_ids[edge.owner_key] = edge_id
            edge_rows.append(
                (
                    edge_id,
                    version_id,
                    edge.edge_type,
                    src_id,
                    dst_id,
                    float(edge.weight),
                    float(edge.confidence),
                    int(bool(edge.visible)),
                    json.dumps(edge.attrs),
                    _now(),
                )
            )

        provenance_rows: list[tuple[Any, ...]] = []
        for item in provenance:
            owner_id = node_ids.get(item.owner_key) if item.owner_kind == "node" else edge_ids.get(item.owner_key)
            if not owner_id:
                continue
            provenance_rows.append(
                (
                    _stable_id(
                        "kgp",
                        version_id,
                        item.owner_kind,
                        owner_id,
                        item.source_kind,
                        item.source_id,
                    ),
                    version_id,
                    item.owner_kind,
                    owner_id,
                    item.source_kind,
                    item.source_id,
                    json.dumps(item.span),
                    json.dumps(item.evidence),
                    _now(),
                )
            )

        with self._lock:
            try:
                if job_id:
                    cursor = self._conn.execute(
                        "UPDATE kg_build_jobs SET stage=stage "
                        "WHERE id=? AND version_id=? AND status='running'",
                        (job_id, version_id),
                    )
                    if not cursor.rowcount:
                        inactive = self._inactive_job_error(
                            job_id,
                            version_id=version_id,
                        )
                        self._conn.rollback()
                        raise inactive
                self._conn.execute("DELETE FROM kg_provenance WHERE version_id=?", (version_id,))
                self._conn.execute("DELETE FROM kg_edges WHERE version_id=?", (version_id,))
                self._conn.execute("DELETE FROM kg_nodes WHERE version_id=?", (version_id,))
                self._conn.executemany(
                    "INSERT INTO kg_nodes"
                    "(id,version_id,node_type,node_key,title,summary,attrs_json,confidence,visible,created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    node_rows,
                )
                self._conn.executemany(
                    "INSERT INTO kg_edges"
                    "(id,version_id,edge_type,src_node_id,dst_node_id,weight,confidence,visible,attrs_json,created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?)",
                    edge_rows,
                )
                self._conn.executemany(
                    "INSERT INTO kg_provenance"
                    "(id,version_id,owner_kind,owner_id,source_kind,source_id,span_json,evidence_json,created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    provenance_rows,
                )
                stats = {
                    "nodes": len(node_rows),
                    "edges": len(edge_rows),
                    "provenance": len(provenance_rows),
                }
                if base_snapshot is None:
                    self._conn.execute(
                        "UPDATE kg_graph_versions SET stats_json=? WHERE id=?",
                        (json.dumps(stats), version_id),
                    )
                else:
                    self._conn.execute(
                        "UPDATE kg_graph_versions SET base_snapshot_json=?, stats_json=? WHERE id=?",
                        (json.dumps(base_snapshot), json.dumps(stats), version_id),
                    )
                self._conn.commit()
                return stats
            except Exception:
                self._conn.rollback()
                raise

    def activate_version(
        self,
        version_id: str,
        *,
        job_id: str | None = None,
        stats: JsonDict | None = None,
    ) -> None:
        ts = _now()
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT workspace_id FROM kg_graph_versions WHERE id=?", (version_id,)
                ).fetchone()
                if not row:
                    raise ValueError(f"unknown knowledge graph version: {version_id}")
                workspace_id = str(row["workspace_id"])
                if job_id:
                    cursor = self._conn.execute(
                        "UPDATE kg_build_jobs "
                        "SET status='succeeded', stage='completed', progress=1.0, finished_at=?, stats_json=? "
                        "WHERE id=? AND version_id=? AND status='running'",
                        (ts, json.dumps(stats or {}), job_id, version_id),
                    )
                    if not cursor.rowcount:
                        inactive = self._inactive_job_error(
                            job_id,
                            version_id=version_id,
                        )
                        self._conn.rollback()
                        raise inactive
                self._conn.execute(
                    "UPDATE kg_graph_versions SET is_active=0 WHERE workspace_id=? AND is_active=1",
                    (workspace_id,),
                )
                self._conn.execute(
                    "UPDATE kg_graph_versions "
                    "SET status='active', is_active=1, finished_at=?, activated_at=? WHERE id=?",
                    (ts, ts, version_id),
                )
                if job_id:
                    self._append_build_job_event(job_id, terminal=True)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def mark_version_failed(
        self,
        version_id: str,
        *,
        stats: JsonDict | None = None,
        job_id: str | None = None,
    ) -> bool:
        with self._lock:
            if job_id:
                cursor = self._conn.execute(
                    "UPDATE kg_build_jobs SET stage=stage "
                    "WHERE id=? AND version_id=? AND status='running'",
                    (job_id, version_id),
                )
                if not cursor.rowcount:
                    self._conn.rollback()
                    return False
            cursor = self._conn.execute(
                "UPDATE kg_graph_versions SET status='failed', is_active=0, finished_at=?, stats_json=? WHERE id=?",
                (_now(), json.dumps(stats or {}), version_id),
            )
            self._conn.commit()
        return bool(cursor.rowcount)

    def create_build_job(self, *, workspace_id: str = "default", backend: str = "thread") -> JsonDict:
        job_id = _uid("kgjob")
        ts = _now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO kg_build_jobs"
                "(id,version_id,workspace_id,backend,backend_job_id,status,stage,progress,started_at,finished_at,error_code,error_message,stats_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (job_id, None, workspace_id, backend, None, "queued", "queued", 0.0, ts, None, None, None, "{}"),
            )
            self._append_build_job_event(job_id, terminal=False)
            self._conn.commit()
        return self.get_build_job(job_id) or {"id": job_id, "status": "queued"}

    def set_build_job_backend_id(self, job_id: str, backend_job_id: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE kg_build_jobs SET backend_job_id=? "
                "WHERE id=? AND status NOT IN ('succeeded','partial','failed')",
                (backend_job_id, job_id),
            )
            self._conn.commit()

    def mark_build_job_running(
        self,
        job_id: str,
        *,
        version_id: str,
        stage: str = "projecting",
    ) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE kg_build_jobs SET version_id=?, status='running', stage=?, progress=? "
                "WHERE id=? AND status NOT IN ('succeeded','partial','failed')",
                (version_id, stage, 0.1, job_id),
            )
            if cursor.rowcount:
                self._append_build_job_event(job_id, terminal=False)
            self._conn.commit()
        return bool(cursor.rowcount)

    def mark_build_job_succeeded(
        self,
        job_id: str,
        *,
        stats: JsonDict | None = None,
    ) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE kg_build_jobs "
                "SET status='succeeded', stage='completed', progress=1.0, finished_at=?, stats_json=? "
                "WHERE id=? AND status NOT IN ('succeeded','partial','failed')",
                (_now(), json.dumps(stats or {}), job_id),
            )
            if cursor.rowcount:
                self._append_build_job_event(job_id, terminal=True)
            self._conn.commit()
        return bool(cursor.rowcount)

    def mark_build_job_failed(
        self,
        job_id: str,
        *,
        error_code: str = "INTERNAL_ERROR",
        error_message: str = "Knowledge graph rebuild failed.",
        stats: JsonDict | None = None,
    ) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE kg_build_jobs "
                "SET status='failed', stage='failed', progress=1.0, finished_at=?, "
                "error_code=?, error_message=?, stats_json=? "
                "WHERE id=? AND status NOT IN ('succeeded','partial','failed')",
                (_now(), error_code, error_message, json.dumps(stats or {}), job_id),
            )
            if cursor.rowcount:
                self._append_build_job_event(job_id, terminal=True)
            self._conn.commit()
        return bool(cursor.rowcount)

    def get_build_job(self, job_id: str) -> JsonDict | None:
        row = self._conn.execute("SELECT * FROM kg_build_jobs WHERE id=?", (job_id,)).fetchone()
        return _decode(row)

    def list_build_job_events(
        self,
        job_id: str,
        *,
        after_seq: int = 0,
    ) -> list[JsonDict]:
        with self._lock:
            return list_job_events(
                self._conn,
                job_kind="graph",
                job_id=job_id,
                after_seq=after_seq,
            )

    def fail_stale_orphan_build_jobs(
        self,
        *,
        workspace_id: str = "default",
        older_than_s: float = STALE_QUEUED_JOB_SECONDS,
    ) -> int:
        cutoff = _now() - max(float(older_than_s), 0.0)
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM kg_build_jobs "
                "WHERE workspace_id=? AND status='queued' "
                "AND backend_job_id IS NULL AND started_at<=? ORDER BY id",
                (workspace_id, cutoff),
            ).fetchall()
            changed = 0
            for row in rows:
                job_id = str(row["id"])
                cursor = self._conn.execute(
                    "UPDATE kg_build_jobs "
                    "SET status='failed', stage='failed', progress=1.0, finished_at=?, "
                    "error_code='ENQUEUE_FAILED', "
                    "error_message='Knowledge graph rebuild could not be queued. Please retry.' "
                    "WHERE id=? AND status='queued'",
                    (_now(), job_id),
                )
                if cursor.rowcount:
                    changed += 1
                    self._append_build_job_event(job_id, terminal=True)
            self._conn.commit()
        return changed

    def _append_build_job_event(self, job_id: str, *, terminal: bool) -> int | None:
        row = self._conn.execute(
            "SELECT * FROM kg_build_jobs WHERE id=?",
            (job_id,),
        ).fetchone()
        job = _decode(row)
        if job is None:
            return None
        return append_job_event(
            self._conn,
            job_kind="graph",
            job=job,
            terminal=terminal,
        )

    def get_running_build_job(self, workspace_id: str = "default") -> JsonDict | None:
        placeholders = ",".join("?" for _ in ACTIVE_STATUSES)
        rows = self._conn.execute(
            f"SELECT * FROM kg_build_jobs WHERE workspace_id=? AND status IN ({placeholders}) "
            "ORDER BY started_at DESC LIMIT 1",
            (workspace_id, *sorted(ACTIVE_STATUSES)),
        ).fetchall()
        return _decode(rows[0]) if rows else None

    def get_active_version(self, workspace_id: str = "default") -> JsonDict | None:
        row = self._conn.execute(
            "SELECT * FROM kg_graph_versions WHERE workspace_id=? AND is_active=1 LIMIT 1",
            (workspace_id,),
        ).fetchone()
        return _decode(row)

    def get_active_status(self, workspace_id: str = "default") -> JsonDict:
        active = self.get_active_version(workspace_id)
        job_row = self._conn.execute(
            "SELECT * FROM kg_build_jobs WHERE workspace_id=? ORDER BY started_at DESC LIMIT 1",
            (workspace_id,),
        ).fetchone()
        counts = {"nodes": 0, "edges": 0}
        if active:
            version_id = str(active["id"])
            node_row = self._conn.execute(
                "SELECT COUNT(*) AS count FROM kg_nodes WHERE version_id=?", (version_id,)
            ).fetchone()
            edge_row = self._conn.execute(
                "SELECT COUNT(*) AS count FROM kg_edges WHERE version_id=?", (version_id,)
            ).fetchone()
            counts = {
                "nodes": int(node_row["count"] if node_row else 0),
                "edges": int(edge_row["count"] if edge_row else 0),
            }
        return {
            "workspace_id": workspace_id,
            "active_version": active,
            "last_build_job": _decode(job_row),
            "counts": counts,
        }

    def get_bounded_graph(
        self,
        *,
        workspace_id: str = "default",
        limit: int = 300,
        edge_types: list[str] | None = None,
    ) -> JsonDict:
        active = self.get_active_version(workspace_id)
        if not active:
            return {"workspace_id": workspace_id, "active_version": None, "nodes": [], "edges": []}
        version_id = str(active["id"])
        bounded_limit = max(1, min(int(limit or 300), 1000))
        concept_limit = min(64, bounded_limit)
        concept_rows = self._conn.execute(
            "SELECT * FROM kg_nodes WHERE version_id=? AND visible=1 AND node_type='concept' "
            "ORDER BY title ASC, node_key ASC LIMIT ?",
            (version_id, concept_limit),
        ).fetchall()
        non_concept_rows = self._conn.execute(
            "SELECT * FROM kg_nodes WHERE version_id=? AND visible=1 AND node_type<>'concept' "
            "ORDER BY node_type ASC, title ASC, node_key ASC LIMIT ?",
            (version_id, bounded_limit - len(concept_rows)),
        ).fetchall()
        node_rows = [*non_concept_rows, *concept_rows]
        nodes = [_public_node(row) for row in node_rows]
        node_ids = [node["id"] for node in nodes]
        if not node_ids:
            return {"workspace_id": workspace_id, "active_version": active, "nodes": [], "edges": []}

        placeholders = ",".join("?" for _ in node_ids)
        params: list[Any] = [version_id, *node_ids, *node_ids]
        edge_filter = ""
        clean_edge_types = [str(item) for item in edge_types or [] if str(item)]
        if clean_edge_types:
            type_placeholders = ",".join("?" for _ in clean_edge_types)
            edge_filter = f" AND edge_type IN ({type_placeholders})"
            params.extend(clean_edge_types)
        edge_rows = self._conn.execute(
            "SELECT * FROM kg_edges WHERE version_id=? AND visible=1 "
            f"AND src_node_id IN ({placeholders}) AND dst_node_id IN ({placeholders})"
            f"{edge_filter} ORDER BY edge_type ASC, src_node_id ASC, dst_node_id ASC",
            params,
        ).fetchall()
        return {
            "workspace_id": workspace_id,
            "active_version": active,
            "nodes": nodes,
            "edges": [_public_edge(row) for row in edge_rows],
        }

    def get_node_detail(self, node_id: str) -> JsonDict | None:
        node_row = self._conn.execute("SELECT * FROM kg_nodes WHERE id=?", (node_id,)).fetchone()
        if not node_row:
            return None
        node = _public_node(node_row)
        edge_rows = self._conn.execute(
            "SELECT * FROM kg_edges WHERE src_node_id=? OR dst_node_id=? ORDER BY edge_type ASC",
            (node_id, node_id),
        ).fetchall()
        provenance_rows = self._conn.execute(
            "SELECT * FROM kg_provenance WHERE owner_kind='node' AND owner_id=? ORDER BY created_at DESC LIMIT 50",
            (node_id,),
        ).fetchall()
        node["edges"] = [_public_edge(row) for row in edge_rows]
        node["provenance"] = [_public_provenance(row) for row in provenance_rows]
        return node


def _public_node(row: Any) -> JsonDict:
    data = _decode(row) or {}
    return {
        "id": data.get("id"),
        "type": data.get("node_type"),
        "key": data.get("node_key"),
        "title": data.get("title") or "",
        "summary": data.get("summary") or "",
        "attrs": data.get("attrs_json") or {},
        "confidence": data.get("confidence"),
        "visible": data.get("visible"),
    }


def _public_edge(row: Any) -> JsonDict:
    data = _decode(row) or {}
    return {
        "id": data.get("id"),
        "type": data.get("edge_type"),
        "source": data.get("src_node_id"),
        "target": data.get("dst_node_id"),
        "weight": data.get("weight"),
        "confidence": data.get("confidence"),
        "visible": data.get("visible"),
        "attrs": data.get("attrs_json") or {},
    }


def _public_provenance(row: Any) -> JsonDict:
    data = _decode(row) or {}
    return {
        "id": data.get("id"),
        "owner_kind": data.get("owner_kind"),
        "source_kind": data.get("source_kind"),
        "source_id": data.get("source_id"),
        "span": data.get("span_json") or {},
        "evidence": data.get("evidence_json") or {},
        "created_at": data.get("created_at"),
    }
