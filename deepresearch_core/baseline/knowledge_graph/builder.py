"""Versioned builder for the research knowledge graph."""

from __future__ import annotations

from typing import Any

from .models import JsonDict
from .projector import KnowledgeGraphProjector
from .store_ops import KnowledgeGraphJobInactive, KnowledgeGraphStore


class KnowledgeGraphBuilder:
    """Run a staging projection and atomically activate it on success."""

    def __init__(self, store: Any, *, projector: KnowledgeGraphProjector | None = None) -> None:
        self._ops = KnowledgeGraphStore(store)
        self._projector = projector or KnowledgeGraphProjector(store)

    def current_base_snapshot(self, *, workspace_id: str = "default") -> JsonDict:
        """Return the current canonical evidence signature without persisting a version."""

        return dict(self._projector.project(workspace_id=workspace_id).base_snapshot)

    def rebuild(self, *, workspace_id: str = "default", job_id: str | None = None) -> JsonDict:
        graph = None
        if job_id:
            version_id = self._ops.begin_build_job_version(
                job_id,
                workspace_id=workspace_id,
                base_snapshot={"workspace_id": workspace_id},
            )
        else:
            version_id = self._ops.create_staging_version(
                workspace_id=workspace_id,
                base_snapshot={"workspace_id": workspace_id},
            )
        try:
            graph = self._projector.project(workspace_id=workspace_id)
            stats = self._ops.replace_version_contents(
                version_id,
                nodes=graph.nodes,
                edges=graph.edges,
                provenance=graph.provenance,
                base_snapshot=graph.base_snapshot,
                job_id=job_id,
            )
            stats.update(graph.stats)
            self._ops.activate_version(version_id, job_id=job_id, stats=stats)
            return {
                "version_id": version_id,
                "status": "succeeded",
                "workspace_id": workspace_id,
                "stats": stats,
            }
        except KnowledgeGraphJobInactive:
            raise
        except Exception:
            stats = graph.stats if graph is not None else {}
            version_changed = self._ops.mark_version_failed(
                version_id,
                stats=stats,
                job_id=job_id,
            )
            if job_id:
                if version_changed:
                    self._ops.mark_build_job_failed(
                        job_id,
                        error_code="INTERNAL_ERROR",
                        error_message="Knowledge graph rebuild failed.",
                        stats=stats,
                    )
            raise
