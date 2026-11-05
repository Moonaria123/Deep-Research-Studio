"""Session-centered view model for the research knowledge graph."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from .models import JsonDict
from .store_ops import KnowledgeGraphStore, _public_edge, _public_node


SESSION_MAP_LAYOUT_MODE = "session_tree"
SESSION_EDGE_TYPES = {
    "session_has_report": "reports",
    "session_used_report": "reports",
    "session_has_note": "notes",
    "session_used_note": "notes",
    "session_used_source": "source_set",
    "session_wrote_memory": "memory_set",
    "session_used_memory": "memory_set",
}


class SessionMapBuilder:
    """Build a complete Session Wiki Map view from the active KG version."""

    def __init__(self, store: Any) -> None:
        self._ops = KnowledgeGraphStore(store)
        self._conn = store._conn

    def build(
        self,
        *,
        workspace_id: str = "default",
        session_limit: int = 50,
        member_preview: int = 0,
    ) -> JsonDict:
        workspace_id = str(workspace_id or "default").strip() or "default"
        active = self._ops.get_active_version(workspace_id)
        if not active:
            return _empty_map(workspace_id)

        version_id = str(active["id"])
        nodes = self._load_nodes(version_id)
        edges = self._load_edges(version_id)
        node_by_id = {str(node["id"]): node for node in nodes}
        node_by_key = {str(node["key"]): dict(node) for node in nodes}
        session_nodes = [node for node in nodes if node.get("type") == "session"]
        session_times = self._session_times(session_nodes)
        ordered_sessions = sorted(
            session_nodes,
            key=lambda node: (
                -session_times.get(_session_id(node), 0.0),
                str(node.get("title") or ""),
                str(node.get("key") or ""),
            ),
        )[: _bounded_limit(session_limit)]

        lanes = {
            str(session["id"]): _new_lane(session)
            for session in ordered_sessions
        }
        referencing_by_memory: dict[str, dict[str, JsonDict]] = defaultdict(dict)
        lane_edges: list[JsonDict] = []
        preview_limit = max(int(member_preview or 0), 0)

        for edge in edges:
            src = node_by_id.get(str(edge.get("source")))
            dst = node_by_id.get(str(edge.get("target")))
            if not src or not dst or src.get("type") != "session":
                continue
            lane = lanes.get(str(src.get("id")))
            if lane is None:
                continue
            bucket = SESSION_EDGE_TYPES.get(str(edge.get("type") or ""))
            if not bucket:
                continue
            lane_edges.append(edge)
            if bucket in {"reports", "notes"}:
                _append_unique_node(lane[bucket], dst)
            else:
                _append_unique_key(lane[bucket]["node_keys"], str(dst.get("key")))
            if bucket == "memory_set":
                referencing_by_memory[str(dst.get("key"))][str(src.get("key"))] = _session_ref(src)

        for lane in lanes.values():
            for bucket in ("source_set", "memory_set"):
                keys = lane[bucket]["node_keys"]
                lane[bucket]["count"] = len(keys)
                lane[bucket]["members"] = _members(keys, node_by_key, preview_limit)
            lane["counts"] = {
                "reports": len(lane["reports"]),
                "notes": len(lane["notes"]),
                "sources": int(lane["source_set"]["count"]),
                "memories": int(lane["memory_set"]["count"]),
            }

        for memory_key, refs in referencing_by_memory.items():
            node = node_by_key.get(memory_key)
            if node is not None:
                node["referencing_sessions"] = sorted(
                    refs.values(),
                    key=lambda item: str(item.get("title") or ""),
                )

        used_keys = _used_node_keys(lanes.values())
        nodes_by_key = {key: node_by_key[key] for key in used_keys if key in node_by_key}
        return {
            "workspace_id": workspace_id,
            "active_version": active,
            "layout_mode": SESSION_MAP_LAYOUT_MODE,
            "session_limit": _bounded_limit(session_limit),
            "member_preview": preview_limit,
            "sessions": list(lanes.values()),
            "nodes_by_key": nodes_by_key,
            "edges": lane_edges,
        }

    def _load_nodes(self, version_id: str) -> list[JsonDict]:
        rows = self._conn.execute(
            "SELECT * FROM kg_nodes WHERE version_id=? AND visible=1 ORDER BY node_type ASC, title ASC, node_key ASC",
            (version_id,),
        ).fetchall()
        return [_public_node(row) for row in rows]

    def _load_edges(self, version_id: str) -> list[JsonDict]:
        rows = self._conn.execute(
            "SELECT * FROM kg_edges WHERE version_id=? AND visible=1 ORDER BY edge_type ASC, src_node_id ASC, dst_node_id ASC",
            (version_id,),
        ).fetchall()
        return [_public_edge(row) for row in rows]

    def _session_times(self, sessions: list[JsonDict]) -> dict[str, float]:
        session_ids = [_session_id(node) for node in sessions if _session_id(node)]
        if not session_ids:
            return {}
        placeholders = ",".join("?" for _ in session_ids)
        rows = self._conn.execute(
            f"SELECT id, created_at, updated_at FROM sessions WHERE id IN ({placeholders})",
            session_ids,
        ).fetchall()
        return {
            str(row["id"]): float(row["updated_at"] or row["created_at"] or 0.0)
            for row in rows
        }


def _empty_map(workspace_id: str) -> JsonDict:
    return {
        "workspace_id": workspace_id,
        "active_version": None,
        "layout_mode": SESSION_MAP_LAYOUT_MODE,
        "session_limit": 0,
        "member_preview": 0,
        "sessions": [],
        "nodes_by_key": {},
        "edges": [],
    }


def _bounded_limit(value: int) -> int:
    return max(1, min(int(value or 50), 200))


def _session_id(node: JsonDict) -> str:
    attrs = node.get("attrs") if isinstance(node.get("attrs"), dict) else {}
    return str(attrs.get("session_id") or str(node.get("key") or "").removeprefix("session:"))


def _session_ref(node: JsonDict) -> JsonDict:
    attrs = node.get("attrs") if isinstance(node.get("attrs"), dict) else {}
    return {
        "node_key": node.get("key"),
        "session_id": attrs.get("session_id") or _session_id(node),
        "title": node.get("title") or "",
        "summary": node.get("summary") or "",
        "jump_uri": attrs.get("jump_uri") or "",
        "assistant_message_id": attrs.get("assistant_message_id") or "",
        "assistant_created_at": attrs.get("assistant_created_at") or "",
    }


def _new_lane(session: JsonDict) -> JsonDict:
    attrs = session.get("attrs") if isinstance(session.get("attrs"), dict) else {}
    session_node = dict(session)
    session_node["assistant_summary"] = attrs.get("assistant_summary") or session.get("summary") or ""
    session_node["assistant_excerpt"] = attrs.get("assistant_excerpt") or session_node["assistant_summary"]
    session_node["last_assistant_message_id"] = attrs.get("assistant_message_id") or ""
    session_node["last_assistant_at"] = attrs.get("assistant_created_at") or ""
    return {
        "session": session_node,
        "reports": [],
        "notes": [],
        "source_set": {"count": 0, "node_keys": [], "members": []},
        "memory_set": {"count": 0, "node_keys": [], "members": []},
        "counts": {"reports": 0, "notes": 0, "sources": 0, "memories": 0},
    }


def _append_unique_node(target: list[JsonDict], node: JsonDict) -> None:
    if any(item.get("key") == node.get("key") for item in target):
        return
    target.append(dict(node))


def _append_unique_key(target: list[str], key: str) -> None:
    if key and key not in target:
        target.append(key)


def _members(keys: list[str], node_by_key: dict[str, JsonDict], preview_limit: int) -> list[JsonDict]:
    visible_keys = keys if preview_limit <= 0 else keys[:preview_limit]
    return [dict(node_by_key[key]) for key in visible_keys if key in node_by_key]


def _used_node_keys(lanes: Any) -> list[str]:
    keys: list[str] = []
    for lane in lanes:
        _append_unique_key(keys, str(lane["session"].get("key")))
        for bucket in ("reports", "notes"):
            for node in lane[bucket]:
                _append_unique_key(keys, str(node.get("key")))
        for bucket in ("source_set", "memory_set"):
            for key in lane[bucket]["node_keys"]:
                _append_unique_key(keys, str(key))
    return keys
