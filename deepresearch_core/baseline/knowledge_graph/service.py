"""Service facade for knowledge graph status, rebuild, graph and node reads."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..errors import BaselineError, ErrorCode
from .builder import KnowledgeGraphBuilder
from .jobs import create_knowledge_graph_job_backend
from .session_map import SessionMapBuilder
from .store_ops import KnowledgeGraphJobInactive, KnowledgeGraphStore


JsonDict = dict[str, Any]


class KnowledgeGraphService:
    """High-level API over versioned KG storage and rebuild scheduling."""

    def __init__(self, store: Any, *, on_rebuild_succeeded: Callable[[JsonDict], None] | None = None) -> None:
        self._store = store
        self._ops = KnowledgeGraphStore(store)
        self._builder = KnowledgeGraphBuilder(store)
        self._session_map = SessionMapBuilder(store)
        self._on_rebuild_succeeded = on_rebuild_succeeded
        self._job_backend = create_knowledge_graph_job_backend(job_fn=self._run_rebuild_safe)

    def start_rebuild(self, payload: JsonDict | None = None) -> JsonDict:
        data = payload if isinstance(payload, dict) else {}
        workspace_id = str(data.get("workspace_id") or "default").strip() or "default"
        self._ops.fail_stale_orphan_build_jobs(workspace_id=workspace_id)
        running = self._ops.get_running_build_job(workspace_id)
        if running:
            raise BaselineError(
                ErrorCode.KNOWLEDGE_GRAPH_BUILD_RUNNING,
                "knowledge graph rebuild is already running",
                details={"job_id": running.get("id"), "workspace_id": workspace_id},
            )
        job = self._ops.create_build_job(workspace_id=workspace_id, backend=self._job_backend.name)
        job_id = str(job["id"])
        try:
            backend_job_id = self._job_backend.enqueue({"workspace_id": workspace_id}, job_id)
        except Exception as error:
            self._ops.mark_build_job_failed(
                job_id,
                error_code="ENQUEUE_FAILED",
                error_message="Knowledge graph rebuild could not be queued. Please retry.",
            )
            raise BaselineError(
                ErrorCode.INTERNAL_ERROR,
                "knowledge graph rebuild could not be queued",
                details={"job_id": job_id, "workspace_id": workspace_id, "error_code": "ENQUEUE_FAILED"},
            ) from error
        self._ops.set_build_job_backend_id(job_id, backend_job_id)
        return self._ops.get_build_job(str(job["id"])) or job

    def get_status(self, workspace_id: str = "default") -> JsonDict:
        return self._ops.get_active_status(workspace_id or "default")

    def get_rebuild_job(
        self,
        job_id: str,
        *,
        workspace_id: str = "default",
    ) -> JsonDict:
        """Return one rebuild job without exposing another workspace."""

        clean_job_id = str(job_id or "").strip()
        expected_workspace = str(workspace_id or "default").strip() or "default"
        job = self._ops.get_build_job(clean_job_id)
        if job is None or str(job.get("workspace_id") or "") != expected_workspace:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "knowledge graph rebuild job not found",
                details={"job_id": clean_job_id},
            )
        return job

    def list_rebuild_job_events(
        self,
        job_id: str,
        *,
        workspace_id: str = "default",
        after_seq: int = 0,
    ) -> list[JsonDict]:
        """Return cursor-bounded events after validating workspace ownership."""

        self.get_rebuild_job(job_id, workspace_id=workspace_id)
        return self._ops.list_build_job_events(job_id, after_seq=after_seq)

    def freeze_retrieval_snapshot(
        self,
        *,
        workspace_id: str = "default",
        requested: bool,
        mode: str,
    ) -> JsonDict:
        """Freeze whether the active graph matches the current canonical evidence surface."""

        snapshot: JsonDict = {
            "requested": bool(requested),
            "effective": False,
            "workspace_id": str(workspace_id or "default"),
            "version_id": "",
            "canonical_evidence_signature": "",
            "reason_codes": [],
        }
        if not requested:
            return snapshot
        if str(mode or "").strip().lower() not in {"multiple", "deep", "council"}:
            snapshot["reason_codes"] = ["mode_not_supported"]
            return snapshot
        active = self._ops.get_active_version(workspace_id or "default")
        if active is None:
            snapshot["reason_codes"] = ["snapshot_unavailable"]
            return snapshot
        snapshot["version_id"] = str(active.get("id") or "")
        if str(active.get("status") or "") != "active":
            snapshot["reason_codes"] = ["graph_not_ready"]
            return snapshot
        stored = active.get("base_snapshot_json")
        stored = stored if isinstance(stored, dict) else {}
        stored_signature = str(stored.get("canonical_evidence_signature") or "")
        if not stored_signature:
            snapshot["reason_codes"] = ["snapshot_signature_unavailable"]
            return snapshot
        try:
            current = self._builder.current_base_snapshot(
                workspace_id=workspace_id or "default"
            )
        except Exception:  # noqa: BLE001 - KG remains an optional retrieval path
            snapshot["reason_codes"] = ["snapshot_read_failed"]
            return snapshot
        current_signature = str(
            current.get("canonical_evidence_signature") or ""
        )
        snapshot["canonical_evidence_signature"] = current_signature
        if not current_signature:
            snapshot["reason_codes"] = ["snapshot_signature_unavailable"]
            return snapshot
        if current_signature != stored_signature:
            snapshot["reason_codes"] = ["snapshot_signature_mismatch"]
            return snapshot
        snapshot["effective"] = True
        return snapshot

    def get_graph(
        self,
        *,
        workspace_id: str = "default",
        limit: int = 300,
        edge_types: list[str] | None = None,
    ) -> JsonDict:
        return self._ops.get_bounded_graph(
            workspace_id=workspace_id or "default",
            limit=limit,
            edge_types=edge_types,
        )

    def get_session_map(
        self,
        *,
        workspace_id: str = "default",
        session_limit: int = 50,
        member_preview: int = 0,
    ) -> JsonDict:
        """Return the Session-centered active KG view model."""

        return self._session_map.build(
            workspace_id=workspace_id or "default",
            session_limit=session_limit,
            member_preview=member_preview,
        )

    def get_node(self, node_id: str) -> JsonDict:
        node = self._ops.get_node_detail(str(node_id or ""))
        if node is None:
            raise BaselineError(
                ErrorCode.INVALID_REQUEST,
                "knowledge graph node not found",
                details={"node_id": str(node_id or "")},
            )
        return {"node": node}

    def mark_rebuild_failed(self, job_id: str, *, message: str = "Knowledge graph rebuild failed.") -> None:
        clean_job_id = _normalize_backend_job_id(job_id)
        self._ops.mark_build_job_failed(
            clean_job_id,
            error_code="INTERNAL_ERROR",
            error_message=message[:240],
        )

    def rebuild_now(self, payload: JsonDict | None = None, *, job_id: str | None = None) -> JsonDict:
        data = payload if isinstance(payload, dict) else {}
        workspace_id = str(data.get("workspace_id") or "default").strip() or "default"
        result = self._builder.rebuild(workspace_id=workspace_id, job_id=job_id)
        self._notify_rebuild_succeeded(result)
        return result

    def _run_rebuild_safe(self, payload: JsonDict, job_id: str) -> None:
        workspace_id = str((payload or {}).get("workspace_id") or "default").strip() or "default"
        try:
            result = self._builder.rebuild(workspace_id=workspace_id, job_id=job_id)
            self._notify_rebuild_succeeded(result)
        except KnowledgeGraphJobInactive as error:
            if not error.benign:
                self._ops.mark_build_job_failed(
                    job_id,
                    error_code="JOB_OWNERSHIP_LOST",
                    error_message="Knowledge graph rebuild lost ownership of its active job.",
                )
            return
        except Exception:
            self._ops.mark_build_job_failed(
                job_id,
                error_code="INTERNAL_ERROR",
                error_message="Knowledge graph rebuild failed.",
            )
            return

    def _notify_rebuild_succeeded(self, result: JsonDict) -> None:
        if self._on_rebuild_succeeded is None:
            return
        try:
            self._on_rebuild_succeeded(result)
        except Exception:
            return


def _normalize_backend_job_id(job_id: str) -> str:
    clean_job_id = str(job_id or "")
    for prefix in ("kg-rebuild:", "kg-rebuild-"):
        if clean_job_id.startswith(prefix):
            return clean_job_id[len(prefix) :]
    return clean_job_id
