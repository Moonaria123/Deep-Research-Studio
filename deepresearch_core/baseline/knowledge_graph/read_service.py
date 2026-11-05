"""Bounded, observe-only reads over the active knowledge graph projection."""

from __future__ import annotations

import time
from collections import deque
from functools import wraps
from typing import Any, Callable, Iterable

from .builder import KnowledgeGraphBuilder
from .read_support import (
    bounded_provenance,
    escape_like,
    public_edge,
    public_node,
    public_version,
)
from .read_status import read_knowledge_graph_status
from .store_ops import KnowledgeGraphStore


JsonDict = dict[str, Any]
Clock = Callable[[], float]

KG_NODE_TYPES = frozenset({"session", "source", "report", "note", "memory", "concept"})
KG_READ_BUDGET_S = 5.0


def _serialized_read(method):
    """Serialize access to the Store's shared SQLite connection."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapped


class KnowledgeGraphReadService:
    """Read one server-owned workspace without creating jobs or graph versions."""

    def __init__(
        self,
        store: Any,
        *,
        workspace_id: str = "default",
        read_budget_s: float = KG_READ_BUDGET_S,
        clock: Clock = time.monotonic,
    ) -> None:
        self._store = store
        self._ops = KnowledgeGraphStore(store)
        self._builder = KnowledgeGraphBuilder(store)
        self._conn = store._conn
        self._lock = store._lock
        self.workspace_id = str(workspace_id or "default")
        self.read_budget_s = max(0.1, min(float(read_budget_s), KG_READ_BUDGET_S))
        self._clock = clock

    @_serialized_read
    def search_nodes(
        self,
        query: str,
        *,
        node_types: Iterable[str] = (),
        session_id: str | None = None,
        limit: int = 20,
    ) -> JsonDict:
        """Search visible nodes in the active version using deterministic lexical ranking."""

        deadline = self._deadline()
        active = self._active_version()
        if active is None:
            return {
                "active_version": None,
                "nodes": [],
                "session_found": None if session_id is None else False,
                "truncated": False,
                "reason_codes": ["kg_active_version_unavailable"],
            }
        self._check(deadline)
        version_id = str(active["id"])
        clean_types = tuple(dict.fromkeys(str(item) for item in node_types if str(item)))
        bounded_limit = max(1, min(int(limit), 50))
        query_text = " ".join(str(query or "").split()).lower()
        contains = f"%{escape_like(query_text)}%"
        prefix = f"{escape_like(query_text)}%"
        clauses = [
            "n.version_id=?",
            "n.visible=1",
            "(LOWER(n.title) LIKE ? ESCAPE '\\' "
            "OR LOWER(n.summary) LIKE ? ESCAPE '\\' "
            "OR LOWER(n.node_key) LIKE ? ESCAPE '\\' "
            "OR LOWER(n.attrs_json) LIKE ? ESCAPE '\\')",
        ]
        params: list[Any] = [version_id, contains, contains, contains, contains]
        if clean_types:
            placeholders = ",".join("?" for _ in clean_types)
            clauses.append(f"n.node_type IN ({placeholders})")
            params.extend(clean_types)

        session_found: bool | None = None
        if session_id is not None:
            session_key = f"session:{str(session_id)}"
            session_row = self._conn.execute(
                "SELECT id FROM kg_nodes WHERE version_id=? AND visible=1 "
                "AND node_type='session' AND node_key=? LIMIT 1",
                (version_id, session_key),
            ).fetchone()
            session_found = session_row is not None
            if session_row is None:
                return {
                    "active_version": public_version(active),
                    "nodes": [],
                    "session_found": False,
                    "truncated": False,
                    "reason_codes": ["kg_session_not_found"],
                }
            session_node_id = str(session_row["id"])
            clauses.append(
                "(n.id=? OR EXISTS ("
                "SELECT 1 FROM kg_edges e WHERE e.version_id=n.version_id AND e.visible=1 "
                "AND e.edge_type LIKE 'session_%' "
                "AND ((e.src_node_id=? AND e.dst_node_id=n.id) "
                "OR (e.dst_node_id=? AND e.src_node_id=n.id))))"
            )
            params.extend((session_node_id, session_node_id, session_node_id))

        params.extend((query_text, query_text, prefix, prefix, bounded_limit + 1))
        rows = self._conn.execute(
            "SELECT n.* FROM kg_nodes n WHERE "
            + " AND ".join(clauses)
            + " ORDER BY CASE "
            "WHEN LOWER(n.title)=? OR LOWER(n.node_key)=? THEN 0 "
            "WHEN LOWER(n.title) LIKE ? ESCAPE '\\' THEN 1 "
            "WHEN LOWER(n.node_key) LIKE ? ESCAPE '\\' THEN 2 ELSE 3 END, "
            "n.node_type ASC,n.title ASC,n.node_key ASC,n.id ASC LIMIT ?",
            params,
        ).fetchall()
        self._check(deadline)
        truncated = len(rows) > bounded_limit
        nodes = [public_node(row, query=query_text) for row in rows[:bounded_limit]]
        return {
            "active_version": public_version(active),
            "nodes": nodes,
            "session_found": session_found,
            "truncated": truncated,
            "reason_codes": ["kg_node_limit_reached"] if truncated else [],
        }

    @_serialized_read
    def get_node(
        self,
        node_id: str,
        *,
        neighbor_limit: int = 20,
        provenance_limit: int = 20,
    ) -> JsonDict | None:
        """Return one active node with bounded direct neighbors and provenance."""

        deadline = self._deadline()
        active = self._active_version()
        if active is None:
            return None
        version_id = str(active["id"])
        row = self._conn.execute(
            "SELECT * FROM kg_nodes WHERE id=? AND version_id=? AND visible=1 LIMIT 1",
            (str(node_id), version_id),
        ).fetchone()
        if row is None:
            return None
        bounded_neighbors = max(1, min(int(neighbor_limit), 50))
        bounded_provenance_limit = max(1, min(int(provenance_limit), 20))
        edge_rows = self._conn.execute(
            "SELECT e.* FROM kg_edges e "
            "JOIN kg_nodes src ON src.id=e.src_node_id AND src.visible=1 "
            "JOIN kg_nodes dst ON dst.id=e.dst_node_id AND dst.visible=1 "
            "WHERE e.version_id=? AND e.visible=1 "
            "AND (e.src_node_id=? OR e.dst_node_id=?) "
            "ORDER BY e.edge_type ASC,e.src_node_id ASC,e.dst_node_id ASC,e.id ASC LIMIT ?",
            (version_id, str(node_id), str(node_id), bounded_neighbors + 1),
        ).fetchall()
        neighbor_truncated = len(edge_rows) > bounded_neighbors
        selected_edges = edge_rows[:bounded_neighbors]
        neighbor_ids = sorted(
            {
                str(edge["dst_node_id"] if str(edge["src_node_id"]) == str(node_id) else edge["src_node_id"])
                for edge in selected_edges
            }
        )
        neighbor_rows = self._nodes_by_id(version_id, neighbor_ids)
        neighbors = []
        for edge in selected_edges:
            source = str(edge["src_node_id"])
            target = str(edge["dst_node_id"])
            other_id = target if source == str(node_id) else source
            other = neighbor_rows.get(other_id)
            if other is None:
                continue
            neighbors.append(
                {
                    "direction": "outgoing" if source == str(node_id) else "incoming",
                    "edge": public_edge(edge),
                    "node": public_node(other),
                }
            )
        provenance_rows = self._conn.execute(
            "SELECT * FROM kg_provenance WHERE version_id=? AND owner_kind='node' "
            "AND owner_id=? ORDER BY created_at DESC,id ASC LIMIT ?",
            (version_id, str(node_id), bounded_provenance_limit + 1),
        ).fetchall()
        provenance, provenance_truncated = bounded_provenance(
            provenance_rows,
            limit=bounded_provenance_limit,
        )
        self._check(deadline)
        reasons = []
        if neighbor_truncated:
            reasons.append("kg_neighbor_limit_reached")
        if provenance_truncated:
            reasons.append("kg_provenance_limit_reached")
        return {
            "active_version": public_version(active),
            "node": public_node(row),
            "neighbors": neighbors,
            "provenance": provenance,
            "truncated": bool(reasons),
            "reason_codes": reasons,
        }

    @_serialized_read
    def get_subgraph(
        self,
        node_id: str,
        *,
        depth: int = 1,
        node_limit: int = 50,
        edge_limit: int = 100,
        fanout_limit: int = 25,
    ) -> JsonDict | None:
        """Return a deterministic, cycle-safe BFS around one active node."""

        deadline = self._deadline()
        active = self._active_version()
        if active is None:
            return None
        version_id = str(active["id"])
        seed_row = self._conn.execute(
            "SELECT * FROM kg_nodes WHERE id=? AND version_id=? AND visible=1 LIMIT 1",
            (str(node_id), version_id),
        ).fetchone()
        if seed_row is None:
            return None
        bounded_depth = max(1, min(int(depth), 2))
        bounded_nodes = max(1, min(int(node_limit), 100))
        bounded_edges = max(1, min(int(edge_limit), 200))
        bounded_fanout = max(1, min(int(fanout_limit), 25))
        nodes: dict[str, Any] = {str(node_id): seed_row}
        edges: dict[str, Any] = {}
        visited = {str(node_id)}
        queue: deque[tuple[str, int]] = deque([(str(node_id), 0)])
        reasons: list[str] = []

        while queue:
            current_id, current_depth = queue.popleft()
            if current_depth >= bounded_depth:
                continue
            self._check(deadline)
            rows = self._conn.execute(
                "SELECT e.* FROM kg_edges e "
                "JOIN kg_nodes src ON src.id=e.src_node_id AND src.visible=1 "
                "JOIN kg_nodes dst ON dst.id=e.dst_node_id AND dst.visible=1 "
                "WHERE e.version_id=? AND e.visible=1 "
                "AND (e.src_node_id=? OR e.dst_node_id=?) "
                "ORDER BY e.edge_type ASC,e.src_node_id ASC,e.dst_node_id ASC,e.id ASC LIMIT ?",
                (version_id, current_id, current_id, bounded_fanout + 1),
            ).fetchall()
            if len(rows) > bounded_fanout:
                reasons.append("kg_fanout_limit_reached")
            for edge in rows[:bounded_fanout]:
                edge_id = str(edge["id"])
                if edge_id in edges:
                    continue
                if len(edges) >= bounded_edges:
                    reasons.append("kg_edge_limit_reached")
                    break
                source = str(edge["src_node_id"])
                target = str(edge["dst_node_id"])
                other_id = target if source == current_id else source
                if other_id not in nodes:
                    if len(nodes) >= bounded_nodes:
                        reasons.append("kg_node_limit_reached")
                        continue
                    other_rows = self._nodes_by_id(version_id, [other_id])
                    other = other_rows.get(other_id)
                    if other is None:
                        continue
                    nodes[other_id] = other
                edges[edge_id] = edge
                if other_id not in visited:
                    visited.add(other_id)
                    queue.append((other_id, current_depth + 1))
            if len(edges) >= bounded_edges:
                break

        self._check(deadline)
        unique_reasons = list(dict.fromkeys(reasons))
        ordered_nodes = sorted(
            nodes.values(),
            key=lambda item: (
                str(item["node_type"] or ""),
                str(item["node_key"] or ""),
                str(item["id"] or ""),
            ),
        )
        ordered_edges = sorted(
            edges.values(),
            key=lambda item: (
                str(item["edge_type"] or ""),
                str(item["src_node_id"] or ""),
                str(item["dst_node_id"] or ""),
                str(item["id"] or ""),
            ),
        )
        return {
            "active_version": public_version(active),
            "root_node_id": str(node_id),
            "nodes": [public_node(item) for item in ordered_nodes],
            "edges": [public_edge(item) for item in ordered_edges],
            "truncated": bool(unique_reasons),
            "reason_codes": unique_reasons,
        }

    @_serialized_read
    def status(self, *, event_limit: int = 20) -> JsonDict:
        """Observe active projection freshness without triggering a rebuild."""

        deadline = self._deadline()
        return read_knowledge_graph_status(
            ops=self._ops,
            builder=self._builder,
            connection=self._conn,
            workspace_id=self.workspace_id,
            event_limit=event_limit,
            deadline=deadline,
            clock=self._clock,
        )

    def _active_version(self) -> JsonDict | None:
        return self._ops.get_active_version(self.workspace_id)

    def _nodes_by_id(self, version_id: str, node_ids: list[str]) -> dict[str, Any]:
        if not node_ids:
            return {}
        placeholders = ",".join("?" for _ in node_ids)
        rows = self._conn.execute(
            f"SELECT * FROM kg_nodes WHERE version_id=? AND visible=1 AND id IN ({placeholders})",
            (version_id, *node_ids),
        ).fetchall()
        return {str(row["id"]): row for row in rows}

    def _deadline(self) -> float:
        return self._clock() + self.read_budget_s

    def _check(self, deadline: float) -> None:
        if self._clock() > deadline:
            raise TimeoutError("knowledge graph read budget exceeded")
