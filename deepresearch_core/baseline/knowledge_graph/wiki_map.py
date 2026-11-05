"""Read model for the Knowledge Graph Wiki Map view."""

from __future__ import annotations

from typing import Any

from .store_ops import KnowledgeGraphStore
from .wiki_models import JsonDict, WIKI_LAYOUT_MODE
from .wiki_store import KnowledgeGraphWikiStore


class WikiMapBuilder:
    """Build a render-ready Wiki Map DTO from the active wiki snapshot."""

    def __init__(self, store: Any) -> None:
        self._kg = KnowledgeGraphStore(store)
        self._wiki = KnowledgeGraphWikiStore(store)

    def build(self, *, workspace_id: str = "default") -> JsonDict:
        workspace_id = str(workspace_id or "default").strip() or "default"
        active_graph = self._kg.get_active_version(workspace_id)
        snapshot = self._wiki.get_active_snapshot(workspace_id)
        last_job = self._wiki.get_latest_refresh_job(workspace_id)
        if not snapshot:
            return {
                "workspace_id": workspace_id,
                "layout_mode": WIKI_LAYOUT_MODE,
                "active_graph_version": active_graph,
                "active_snapshot": None,
                "stale": bool(active_graph),
                "stale_reason": "no_active_wiki_snapshot" if active_graph else "",
                "stats": {"topics": 0, "tree_edges": 0, "cross_links": 0},
                "topics": [],
                "tree_edges": [],
                "cross_links": [],
                "last_refresh_job": last_job,
            }

        graph_version_id = str(active_graph.get("id") or "") if active_graph else ""
        stale = bool(snapshot.get("stale")) or bool(graph_version_id and snapshot.get("graph_version_id") != graph_version_id)
        stale_reason = str(snapshot.get("stale_reason") or "")
        if not stale_reason and stale:
            stale_reason = "graph_version_changed"
        snapshot_id = str(snapshot["id"])
        topics = self._wiki.get_snapshot_topics(snapshot_id)
        tree_edges = self._wiki.get_snapshot_tree_edges(snapshot_id)
        cross_links = self._wiki.get_snapshot_cross_links(snapshot_id)
        stats = dict(snapshot.get("stats") or {})
        stats.setdefault("topics", len(topics))
        stats.setdefault("tree_edges", len(tree_edges))
        stats.setdefault("cross_links", len(cross_links))
        return {
            "workspace_id": workspace_id,
            "layout_mode": WIKI_LAYOUT_MODE,
            "active_graph_version": active_graph,
            "active_snapshot": snapshot,
            "stale": stale,
            "stale_reason": stale_reason,
            "stats": stats,
            "topics": topics,
            "tree_edges": tree_edges,
            "cross_links": cross_links,
            "last_refresh_job": last_job,
        }
